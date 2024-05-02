# quick sanity check to determine if the GPU is detected. 
import tensorflow as tf

print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
