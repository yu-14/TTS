# prepare_dataset.py

from datasets import load_dataset, Dataset

# Load your existing dataset
dataset_path = "data/dataset/technical_terms.txt"
dataset = load_dataset('text', data_files=dataset_path)

# Create a new dataset with both text and audio_target (mock)
def create_mock_dataset(dataset):
    return Dataset.from_dict({
        'text': dataset['train']['text'],
        'audio_target': [None] * len(dataset['train']['text'])  # Placeholder for audio targets
    })

mock_dataset = create_mock_dataset(dataset)

# Save the mock dataset to disk for later use
mock_dataset.save_to_disk("data/dataset/mock_technical_terms")

print("Mock dataset created and saved.")