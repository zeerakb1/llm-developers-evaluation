# Install necessary libraries
# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip install -q -U pandas datasets scipy ipywidgets openpyxl

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from datetime import datetime

# Load StackExchange dataset from Excel
def load_qa_excel(file_path):
    df = pd.read_excel(file_path)
    question_titles = df['Question Title'].tolist()
    question_bodies = df['Question Body'].tolist()
    accepted_answers = df['Accepted Answer Body'].tolist()
    dataset = [{'input': f"Title: {title}\nBody: {body}", 'output': answer}
               for title, body, answer in zip(question_titles, question_bodies, accepted_answers)]
    return dataset

qa_file = '../../stackExchangeQsAndAnswersTest.xlsx'
stackexchange_data = load_qa_excel(qa_file)

# Define the model and tokenizer
base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
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

# Preprocess and tokenize datasets
tokenized_train_data = [generate_and_tokenize_prompt(data) for data in stackexchange_data]

# Set up model for parameter-efficient fine-tuning (LoRA)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
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

# Setup for multi-GPU and accelerator
fsdp_plugin = FullyShardedDataParallelPlugin()
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
model = accelerator.prepare_model(model)

# Define training arguments
project = "stackexchange-finetune"
run_name = f"{base_model_id}-{project}"
output_dir = "./" + run_name

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=1000,
    learning_rate=2.5e-5,
    logging_steps=50,
    bf16=True,
    optim="paged_adamw_8bit",
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    do_eval=True,
    report_to="wandb",
    run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

# Trainer setup
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_data,
    args=training_args,
    data_collator=data_collator
)

# Train the model
model.config.use_cache = False
trainer.train()

# Save final model for RAG usage
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Load and test the model
ft_model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model_input = tokenizer("Provide a sample question here", return_tensors="pt").to("cuda")
ft_model.eval()

with torch.no_grad():
    output = ft_model.generate(**model_input, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
