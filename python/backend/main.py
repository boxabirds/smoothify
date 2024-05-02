import subprocess
from tqdm import tqdm
import numpy as np
import requests
import tensorflow as tf
import tensorflow_hub as hub
from typing import Generator, Iterable, List
import mediapy as media
import glob
import cv2
import os
import argparse
from pathlib import Path
import time

print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# we get CPU advanced instruction warnings if we don't do this but they're irrelevant for GPUs
# "To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags."
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Inspired by https://www.tensorflow.org/hub/tutorials/tf_hub_film_example

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)


def get_gpu_usage():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'])
        gpu_info = output.decode('utf-8').strip().split(',')
        gpu_utilization = int(gpu_info[0])
        gpu_memory_used = int(gpu_info[1])
        gpu_memory_total = int(gpu_info[2])
        gpu_memory_percentage = (gpu_memory_used / gpu_memory_total) * 100
        return gpu_utilization, gpu_memory_used, gpu_memory_percentage
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving GPU usage: {e}")
        return None, None, None


gpu_utilization_values = []
gpu_memory_used_values = []
gpu_memory_percentage_values = []

def load_image(img_path: Path):
  """Returns an image with shape [height, width, num_channels], with pixels in [0..1] range, and type np.float32."""

  image_data = tf.io.read_file(str(img_path))
  image = tf.io.decode_image(image_data, channels=3)
  image_numpy = tf.cast(image, dtype=tf.float32).numpy()
  return image_numpy / _UINT8_MAX_F
  # TODO apparently this is a better approach
  #image = tf.io.decode_image(image_data, channels=3, expand_animations=False)
  #return image.numpy()

def extract_frames(movie_path: Path) -> tuple[Path, int, int]:
    # Create the frames directory name
    movie_name = movie_path.stem
    frames_dir = Path(f"{movie_name}-frames")
    
    # Create the frames directory if it doesn't exist
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(str(movie_path))
    
    # Get the FPS of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        frame_count = 0
        while frame_count < total_frames:
            # Generate the frame filename
            frame_filename = frames_dir / f"frame_{frame_count:06d}.png"
            
            if not frame_filename.exists():
                # Read the next frame
                success, frame = video.read()
                
                if success:
                    # Save the frame as a PNG image
                    cv2.imwrite(str(frame_filename), frame)
                else:
                    print(f"Failed to read frame {frame_count}")
            
            # Increment the frame counter
            frame_count += 1
            
            # Update the progress bar
            pbar.update(1)
    
    # Release the video file
    video.release()
    
    print(f"{movie_name} runs at {fps}: {frame_count} frames from {movie_path} into {frames_dir}")
    return frames_dir, frame_count, fps



def _pad_to_align(x, align):
  """Pads image batch x so width and height divide by align.

  Args:
    x: Image batch to align.
    align: Number to align to.

  Returns:
    1) An image padded so width % align == 0 and height % align == 0.
    2) A bounding box that can be fed readily to tf.image.crop_to_bounding_box
      to undo the padding.
  """
  # Input checking.
  assert np.ndim(x) == 4
  assert align > 0, 'align must be a positive number.'

  height, width = x.shape[-3:-1]
  height_to_pad = (align - height % align) if height % align != 0 else 0
  width_to_pad = (align - width % align) if width % align != 0 else 0

  bbox_to_pad = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height + height_to_pad,
      'target_width': width + width_to_pad
  }
  padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
  bbox_to_crop = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height,
      'target_width': width
  }
  return padded_x, bbox_to_crop


class Interpolator:
  """A class for generating interpolated frames between two input frames.

  Uses the Film model from TFHub
  """

  def __init__(self, align: int = 64) -> None:
    
    """Loads a saved model.

    Args:
      align: 'If >1, pad the input size so it divides with this before
        inference.'
    """
    model_url = "https://tfhub.dev/google/film/1"
    start_time = time.time()
    print(f"Loading model from {model_url}…")
    self._model = hub.load(model_url)
    end_time = time.time()
    print(f"-- loaded model in {end_time - start_time:1f} seconds")

    # Get the model's cache directory
    # cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "tfhub_modules")

    # # Determine the model file name
    # model_name = model_url.split("/")[-1].replace("-", "_")

    # # Construct the full path to the model directory
    # model_dir = os.path.join(cache_dir, model_name)

    # # Calculate the size of the model directory
    # model_size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)))

    # print(f"Model size: {model_size} bytes")

    self._align = align
    self.last_frame_time = -1

  def __call__(self, x0: np.ndarray, x1: np.ndarray,
               dt: np.ndarray) -> np.ndarray:
    """Generates an interpolated frame between given two batches of frames.

    All inputs should be np.float32 datatype.

    Args:
      x0: First image batch. Dimensions: (batch_size, height, width, channels)
      x1: Second image batch. Dimensions: (batch_size, height, width, channels)
      dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

    Returns:
      The result with dimensions (batch_size, height, width, channels).
    """
    start_time = time.time()
    if self._align is not None:
      x0, bbox_to_crop = _pad_to_align(x0, self._align)
      x1, _ = _pad_to_align(x1, self._align)

    inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
    result = self._model(inputs, training=False)
    image = result['image']

    if self._align is not None:
      image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
    end_time = time.time()
    self.last_frame_time = end_time - start_time
    return image.numpy()
  


def _recursive_generator(
    frame1: np.ndarray, frame2: np.ndarray, num_recursions: int,
    interpolator: Interpolator ) -> Generator[np.ndarray, None, None]:
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input image 1.
    frame2: Input image 2.
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    yield frame1
  else:
    # Adds the batch dimension to all inputs before calling the interpolator,
    # and remove it afterwards.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator(
        np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]
    yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                    interpolator)
    yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                    interpolator)



def interpolate_recursively(
    frames: List[np.ndarray],
    num_recursions: int,
    interpolator: Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    num_recursions: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  for i in range(1, n):
      yield from _recursive_generator(frames[i - 1], frames[i],
                                      num_recursions, interpolator)
  # Separately yield the final frame.
  yield frames[-1]



def generate_video(frames, output_fps:int, output_path:Path, multiplier=2):
    interpolator = Interpolator()
    # assert that multiplier can only be 2 or 4
    assert multiplier in [2, 4], f"multiplier must be 2 or 4, got {multiplier}"

    # interpolation is done recursively. 
    # 2x multiplier means 1 extra frame, recursion depth ONE
    # 4x means 3 extra frames, recursion depth TWO. 
    # TODO this is acceptable for 2 and 4 multipliers ONLY
    num_recursions = multiplier // 2
    with tqdm(total=len(frames)-1, desc="Processing frames", unit="frames") as pbar:
      def update_progress(frame):
          pbar.update(1)
          pbar.set_postfix(frame_time=interpolator.last_frame_time)
          gpu_utilization, gpu_memory_used, gpu_memory_percentage = get_gpu_usage()
          if gpu_utilization is not None and gpu_memory_used is not None and gpu_memory_percentage is not None:
              gpu_utilization_values.append(gpu_utilization)
              gpu_memory_used_values.append(gpu_memory_used)
              gpu_memory_percentage_values.append(gpu_memory_percentage)

          return frame
      
      frames = map(update_progress, interpolate_recursively(frames, num_recursions, interpolator))
      frames = list(frames)

    print(f'Creating {output_path} with {len(frames)} frames at {output_fps} FPS')
    media.write_video(output_path, frames, fps=output_fps)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video frame interpolation tool")
    parser.add_argument("--source", required=True, help="Path to the input video file")
    parser.add_argument("--multiplier", type=int, choices=[2, 4], default=2, help="Multiplier (2 or 4)")
    parser.add_argument("--change", choices=["fps", "duration"], default="fps", help="Change 'fps' or 'duration' (default: fps)")
    parser.add_argument("--dest", help="Path to the output video file (default: <source>-<fps/duration>.mp4)")
    args = parser.parse_args()

    frames_dir, frame_count, fps = extract_frames(Path(args.source))
    output_fps = fps

    if args.change == "fps":
        output_fps *= args.multiplier
    elif args.change == "duration":
        output_fps = fps

    if args.dest:
        output_path = Path(args.dest)
    else:
        source_name = Path(args.source).stem
        if args.change == "fps":
            output_path = Path(f"{source_name}-{output_fps}-fps.mp4")
        elif args.change == "duration":
            output_duration = (frame_count / fps) * args.multiplier
            output_path = Path(f"{source_name}-{output_duration:.2f}s.mp4")

    filenames = sorted(frames_dir.glob("*.png"))
    input_frames = [load_image(image) for image in filenames]

    generate_video(input_frames, output_fps, output_path, args.multiplier, args.change)
    if len(gpu_utilization_values) > 0 and len(gpu_memory_used_values) > 0 and len(gpu_memory_percentage_values) > 0:
        avg_gpu_utilization = sum(gpu_utilization_values) / len(gpu_utilization_values)
        max_gpu_utilization = max(gpu_utilization_values)
        min_gpu_utilization = min(gpu_utilization_values)
        
        avg_gpu_memory_used = sum(gpu_memory_used_values) / len(gpu_memory_used_values)
        max_gpu_memory_used = max(gpu_memory_used_values)
        min_gpu_memory_used = min(gpu_memory_used_values)
        
        avg_gpu_memory_percentage = sum(gpu_memory_percentage_values) / len(gpu_memory_percentage_values)
        max_gpu_memory_percentage = max(gpu_memory_percentage_values)
        min_gpu_memory_percentage = min(gpu_memory_percentage_values)
        
        print(f"\nGPU Utilization: Avg={avg_gpu_utilization:.2f}%, Max={max_gpu_utilization}%, Min={min_gpu_utilization}%")
        print(f"GPU Memory Used: Avg={avg_gpu_memory_used:.2f}MB, Max={max_gpu_memory_used}MB, Min={min_gpu_memory_used}MB")
        print(f"GPU Memory Usage: Avg={avg_gpu_memory_percentage:.2f}%, Max={max_gpu_memory_percentage:.2f}%, Min={min_gpu_memory_percentage:.2f}%")
    else:
        print("No GPU usage data collected.")