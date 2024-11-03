import os
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset

# Load the processor and models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load speaker embeddings (example from CMU Arctic)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)

# Load your dataset
dataset_path = "data/dataset/technical_terms.txt"
try:
    with open(dataset_path, "r") as file:
        data = file.readlines()
except FileNotFoundError:
    print(f"Error: The file {dataset_path} was not found.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)
# Clean up the data
data = [line.strip() for line in data if line.strip()]

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Generate speech for each term in the dataset
for text in data:
    # Prepare inputs for the model with padding and truncation
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)

    # Generate speech with speaker embeddings
    with torch.no_grad():
        mel_spectrogram = model.generate(input_ids=inputs.input_ids, speaker_embeddings=speaker_embeddings)

        # Use vocoder to convert mel-spectrogram to waveform
        waveform = vocoder(mel_spectrogram).squeeze().cpu().numpy()  # Convert to NumPy array

    # Save the generated waveform as a WAV file
    output_file = f"outputs/{text.replace(' ', '_').replace('?', '').replace('.', '')}.wav"
    sf.write(output_file, waveform, samplerate=22050)  # Ensure correct sample rate

print("Speech generation complete!")