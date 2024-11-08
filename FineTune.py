import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datetime import datetime



# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load StackExchange dataset from Excel
def load_qa_excel(file_path):
    df = pd.read_excel(file_path)
    question_bodies = df['Question Body'].tolist()
    accepted_answers = df['Accepted Answer Body'].tolist()
    dataset = [{'input': f"Body: {body}", 'output': answer}
               for body, answer in zip(question_bodies, accepted_answers)]
    return dataset

# File path for dataset
qa_file = './Dataset/DataExcel/stackExchangeQsAndAnswersTrain.xlsx'
stackexchange_data = load_qa_excel(qa_file)

# Split data into training and evaluation sets
train_data = stackexchange_data[0:]  # Use 20,000 rows for training
eval_data = stackexchange_data[:100]   # Use 1,000 rows for evaluation

# Define the model and tokenizer

base_model_id = "upstage/SOLAR-10.7B-Instruct-v1.0"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model on GPU
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config).to(device)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, model_max_length=512, padding_side="left", add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the data
def tokenize(prompt):
    result = tokenizer(prompt, truncation=True, max_length=512, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = f"### Question:\n{data_point['input']}\n### Answer:\n{data_point['output']}\n"
    return tokenize(full_prompt)

# Tokenize each dataset separately
tokenized_train_data = [generate_and_tokenize_prompt(data) for data in train_data]
tokenized_eval_data = [generate_and_tokenize_prompt(data) for data in eval_data]

# Set up model for parameter-efficient fine-tuning (LoRA)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Print trainable parameters
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || Total params: {total_params} || Trainable%: {100 * trainable_params / total_params}")

print_trainable_parameters(model)

# Define training arguments with modified eval dataset
project = "stackexchange-finetune3"
run_name = f"{base_model_id}-{project}"
output_dir = "./" + run_name

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=500,
    learning_rate=2.5e-5,
    logging_steps=500,
    optim="paged_adamw_8bit",
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=100,
    do_eval=True,
    report_to="wandb",
    run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    no_cuda=False  # Enable CUDA for GPU usage
)

# Trainer setup
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_eval_data,  # Add eval dataset here
    args=training_args,
    data_collator=data_collator
)

# Train the model
model.config.use_cache = False
trainer.train()

print("Model Training Done")

# Transfer the model to CPU before saving
model = model.to('cpu')  # Move the model to CPU

# Save final model for RAG usage
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Model Saving Done")