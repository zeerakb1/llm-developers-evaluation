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


query_file = './Dataset/DataExcel/stackExchangeQsAndAnswersTest.xlsx'
query_data = load_qa_excel(query_file)[:5]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

ft_model = AutoModelForCausalLM.from_pretrained(
    output_dir, quantization_config=bnb_config
).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(output_dir)

ft_model.eval()

generated_answers = []

c = 1
for data in query_data:
    print(c)
    c = c + 1
    question_input = f"Title: {data['input']}\n"
    inputs = tokenizer(question_input, return_tensors="pt", truncation=True, max_length=256).to('cuda')
    

    with torch.no_grad():
        with torch.amp.autocast('cuda'): 
            output = ft_model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id)
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_answers.append({"Question": question_input, "Generated Answer": answer})

output_df = pd.DataFrame(generated_answers)
output_df.to_csv('Dataset/DataExcel/generated_answers4bit.csv', index=False)
print("Generated answers saved to 'Dataset/DataExcel/generated_answers4bit.csv'")
