from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, GPT2Tokenizer
from dataset_preparation import load_and_tokenize_dataset

# Load dataset
dataset = load_and_tokenize_dataset()

# Load model
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  

device = "cpu" # or cuda
model.to(device)

training_args = TrainingArguments(
    output_dir="gpt2-finance",
    evaluation_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=500,
    save_total_limit=2,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("gpt2-finance-model-latest")
