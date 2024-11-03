import os
import sounddevice as sd
import soundfile as sf

# Function to play audio with file existence check
def play_audio(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return  # Exit the function if the file is not found

    # If the file exists, read and play the audio
    data, samplerate = sf.read(file_path)
    print(f"Data shape: {data.shape}")  # Print the shape of the audio data

    # If data has more than one dimension, convert to mono by averaging channels
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # Convert stereo to mono by averaging channels

    sd.play(data, samplerate)
    sd.wait()  # Wait until file is done playing

# Example usage
play_audio("outputs/Describe_RESTful_services.wav")  # Replace with any file name you want to test