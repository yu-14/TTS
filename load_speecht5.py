from transformers import SpeechT5ForConditionalGeneration, AutoTokenizer

model_name = "microsoft/speecht5_tts"
model = SpeechT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model and tokenizer loaded successfully!")
