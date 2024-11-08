import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datetime import datetime

output_dir = 'upstage/SOLAR-10.7B-Instruct-v1.0-stackexchange-finetune3'


def load_qa_excel(file_path):
    df = pd.read_excel(file_path)
    question_titles = df['Question Title'].tolist()
    question_bodies = df['Question Body'].tolist()
    accepted_answers = df['Accepted Answer Body'].tolist()
    dataset = [{'input': f"Title: {title}\nBody: {body}", 'output': answer}
               for title, body, answer in zip(question_titles, question_bodies, accepted_answers)]
    return dataset


# Load the query file and generate answers for the first 5 rows
query_file = './Dataset/DataExcel/stackExchangeQsAndAnswersTest.xlsx'
query_data = load_qa_excel(query_file)[:1]  # Load only the first 5 rows for inference

# 4-bit quantization setup using bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,  # Optional: Use double quantization for 4-bit models (can improve performance)
    bnb_4bit_quant_type="nf4",       # Quantization type
    bnb_4bit_compute_dtype=torch.float16  # Match dtype to float16 for better performance
)

# Generate answers using the fine-tuned model with 4-bit quantization
ft_model = AutoModelForCausalLM.from_pretrained(
    output_dir, quantization_config=bnb_config
).to('cuda')  # Load model on GPU
tokenizer = AutoTokenizer.from_pretrained(output_dir)

ft_model.eval()  # Set model to evaluation mode

generated_answers = []

c = 1
for data in query_data:
    print(c)
    c = c + 1
    question_input = f"Title: {data['input']}\n"
    inputs = tokenizer(question_input, return_tensors="pt", truncation=True, max_length=256).to('cuda')  # Tokenize with truncation and max_length
    
    # Use torch.no_grad() to save memory during inference
    with torch.no_grad():
        with torch.amp.autocast('cuda'):  # Updated mixed-precision inference
            output = ft_model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id)
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_answers.append({"Question": question_input, "Generated Answer": answer})

# Save the generated answers to a CSV file
output_df = pd.DataFrame(generated_answers)
output_df.to_csv('Dataset/DataExcel/generated_answers4bit.csv', index=False)
print("Generated answers saved to 'Dataset/DataExcel/generated_answers4bit.csv'")
