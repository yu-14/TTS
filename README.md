# Text-to-Speech (TTS) Project

## Overview
This project focuses on fine-tuning text-to-speech (TTS) models to accurately pronounce technical jargon commonly used in English technical interviews. The objective is to enhance model performance for specific vocabulary while exploring optimization techniques for fast inference and model size reduction through quantization.

## Table of Contents
- [Installation Instructions](#installation-instructions)
- [Usage Instructions](#usage-instructions)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## Installation Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/yu-14/TTS.git
    cd TTS
    ```
2. Set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install required dependencies:
    ```bash
    pip install -r requirements.txt  # Ensure you have a requirements.txt with necessary packages listed.
    ```

## Usage Instructions
To generate speech from technical terms:
```bash
python main.py

This command will read from data/dataset/technical_terms.txt and output audio files in the outputs directory.

Dataset Information
The dataset used in this project is located at data/dataset/technical_terms.txt, which contains a list of technical terms relevant to English technical interviews.

TTS/
│
├── data/
│   └── dataset/
│       └── technical_terms.txt  # Dataset containing technical terms
│
├── models/                      # Directory for model checkpoints (currently empty)
│
├── outputs/                     # Directory for generated audio files
│   ├── <generated_audio_files>   # Generated WAV audio files
│
├── venv/                        # Virtual environment directory
│
├── main.py                     # Main script for TTS generation
├── convert_multiple_pcm_to_wav.py  # Script to convert PCM files to WAV format
├── play_audio.py               # Script to play audio files
├── check_mock_dataset.py       # Script to validate dataset integrity
├── check_transformers.py       # Script to check transformer library installation
├── fine_tune_speecht5.py      # Script for fine-tuning SpeechT5 model
├── load_dataset.py             # Script to load datasets
├── load_speecht5.py           # Script to load SpeechT5 model and processor
├── prepare_dataset.py          # Script to prepare the dataset for training
└── test_import.py              # Script to test library imports

Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

License
This project is licensed under the MIT License.
