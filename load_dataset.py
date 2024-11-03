# load_dataset.py

from datasets import load_dataset

# Load your dataset
dataset_path = "data/dataset/technical_terms.txt"
dataset = load_dataset('text', data_files=dataset_path)

# Optional: Print some examples from the dataset to verify it's loaded correctly
print(dataset)
print("Sample data:", dataset['train'][0])  # Print the first entry in the training set