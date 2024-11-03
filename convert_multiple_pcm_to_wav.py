import os
import numpy as np
import soundfile as sf

# Directory containing your PCM files
pcm_directory = 'C:/Users/pilli uttej/OneDrive/Documents/TTS_Proj/outputs'  # Update this path

# Sample rate and number of channels (adjust as necessary)
sample_rate = 16000  # Set this according to your audio
num_channels = 1      # Set this according to your audio (1 for mono, 2 for stereo)
dtype = 'int16'       # Use 'float32' if your PCM data is in float format

# Iterate through each file in the directory
for filename in os.listdir(pcm_directory):
    if filename.endswith('.pcm'):  # Process only PCM files
        pcm_file_path = os.path.join(pcm_directory, filename)
        
        # Read raw PCM data from a file
        with open(pcm_file_path, 'rb') as f:
            pcm_data = np.frombuffer(f.read(), dtype=dtype)

        # Reshape if stereo (optional)
        if num_channels == 2:
            pcm_data = pcm_data.reshape((-1, 2))

        # Define output WAV file path
        output_wav_path = os.path.join(pcm_directory, f"{os.path.splitext(filename)[0]}.wav")

        # Save as WAV
        sf.write(output_wav_path, pcm_data, sample_rate)

        print(f"Converted {filename} to {os.path.basename(output_wav_path)}")

print("All conversions complete! You can now listen to the WAV files.")