from datasets import load_dataset
from transformers import GPT2Tokenizer

def format_conversation(batch, tokenizer):
    formatted_texts = [f"User: {u}\nAssistant: {a}" for u, a in zip(batch["user"], batch["assistant"])]
    tokens = tokenizer(formatted_texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {"input_ids": tokens["input_ids"].tolist(), "attention_mask": tokens["attention_mask"].tolist()}

def load_and_tokenize_dataset():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    ds = load_dataset("Josephgflowers/Finance-Instruct-500k")
    tokenized_ds = ds.map(lambda batch: format_conversation(batch, tokenizer), batched=True, num_proc=2)
    tokenized_ds = tokenized_ds.remove_columns(["user", "assistant"])
    
    return tokenized_ds
