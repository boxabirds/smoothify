FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y libgl1-mesa-glx zsh ffmpeg

RUN pip install tqdm tensorflow-hub mediapy opencv-python-headless requests

# Set zsh as the default shell
SHELL ["/bin/zsh", "-c"]

# Create a non-root user and switch to it
RUN useradd -m -s /bin/zsh developer

# Create a .zshrc file with a simple configuration
RUN echo "export PROMPT='%n@%m:%~$ '" > /home/developer/.zshrc

# Set the correct permissions for the .zshrc file
RUN chown developer:developer /home/developer/.zshrc

USER developer

# Set the working directory
WORKDIR /workspace

# Start zsh when the container launches
CMD ["/bin/zsh"]
