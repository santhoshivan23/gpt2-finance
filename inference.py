from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_path="gpt2-finance-model-latest"):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cpu" # or cuda
    model.to(device)
    return model, tokenizer, device

def generate_response(prompt, model, tokenizer, device, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

model, tokenizer, device = load_model()
print(generate_response(prompt="What is Long Term Capital Gains in India?", model=model, tokenizer=tokenizer, device=device))