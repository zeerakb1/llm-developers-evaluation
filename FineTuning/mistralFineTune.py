import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Load the StackExchange data from the Excel file
def load_qa_excel(file_path):
    df = pd.read_excel(file_path)
    question_titles = df['Question Title'].tolist()
    question_bodies = df['Question Body'].tolist()
    accepted_answers = df['Accepted Answer Body'].tolist()
    return question_titles, question_bodies, accepted_answers

# Prepare the dataset
def prepare_dataset(qa_file):
    question_titles, question_bodies, accepted_answers = load_qa_excel(qa_file)
    dataset = [{'input': f"Title: {title}\nBody: {body}", 'output': answer}
               for title, body, answer in zip(question_titles, question_bodies, accepted_answers)]
    return dataset

# Tokenizer and Model Setup
model_name = 'mistralai/Mistral-7B-v0.1'  # Replace with Mistral model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configure PEFT with LoRA
peft_config = LoRAConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q_proj', 'v_proj']
)
model = get_peft_model(model, peft_config)

# Tokenize data
def preprocess_data(example):
    inputs = tokenizer(example['input'], return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    labels = tokenizer(example['output'], return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    inputs['labels'] = labels['input_ids']
    return inputs

# Main
if __name__ == "__main__":
    qa_file = '../../stackExchangeQsAndAnswersTest.xlsx'
    output_model_dir = './mistral_finetuned'

    # Load and preprocess the dataset
    dataset = prepare_dataset(qa_file)
    tokenized_data = [preprocess_data(d) for d in dataset]

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        fp16=True if torch.cuda.is_available() else False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        tokenizer=tokenizer
    )

    # Fine-tune the model
    trainer.train()

    # Save the model and tokenizer for later use
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
