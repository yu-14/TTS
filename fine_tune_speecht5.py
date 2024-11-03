import torch
from transformers import SpeechT5Tokenizer, SpeechT5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Custom dataset class
class CustomSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=80):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize labels (ensure labels are also tokenized)
        labels = self.tokenizer(
            label,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Return as a flattened tensor
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels['input_ids'].flatten()
        }

# Example texts and labels
texts = ["Hello, how are you?", "This is a test."]
labels = ["Hello, I'm fine.", "Testing is essential."]

# Initialize tokenizer and model
tokenizer = SpeechT5Tokenizer.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForConditionalGeneration.from_pretrained("microsoft/speecht5_tts")

# Prepare dataset and dataloader
dataset = CustomSpeechDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training parameters
num_epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f'Training Epoch {epoch + 1}'):
        optimizer.zero_grad()

        # Move inputs to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        try:
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            print(f"Input IDs shape: {input_ids.shape}, Attention Mask shape: {attention_mask.shape}, Labels shape: {labels.shape}")
            continue

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # Save model after each epoch
    model.save_pretrained(f'models/fine_tuned_speecht5_epoch{epoch + 1}')
    tokenizer.save_pretrained(f'models/fine_tuned_speecht5_epoch{epoch + 1}')

print("Fine-tuning complete!")
