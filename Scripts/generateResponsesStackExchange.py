import pandas as pd
import subprocess
import time

def run_codellama2(text):
    try:
        result = subprocess.run(['ollama', 'run', 'codellama', text], capture_output=True, text=True, timeout=120)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Response timed out"
    except Exception as e:
        return f"Error: {e}"

def run_starcoder2(text):
    try:
        result = subprocess.run(['ollama', 'run', 'starcoder2', text], capture_output=True, text=True, timeout=120)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Response timed out"
    except Exception as e:
        return f"Error: {e}"

def run_solar(text):
    try:
        result = subprocess.run(['ollama', 'run', 'solar', text], capture_output=True, text=True, timeout=120)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Response timed out"
    except Exception as e:
        return f"Error: {e}"

def run_mistral(text):
    try:
        result = subprocess.run(['ollama', 'run', 'mistral', text], capture_output=True, text=True, timeout=120)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Response timed out"
    except Exception as e:
        return f"Error: {e}"

def process_excel_file(file_path, num_rows):
    try:
        df = pd.read_excel(file_path)
        df_to_process = df.head(num_rows)

        df['codellama_response'] = ""
        df['starcoder2_response'] = ""
        df['solar_response'] = ""
        df['mistral_response'] = ""

        for index, row in df_to_process.iterrows():
            question_title = row['Question Title']
            question_body = row['Question Body']
            combined_question = f"{question_title} {question_body}"
            combined_text = f"Following is a question posted on a forum, generate a helpful response that is less than 200 words: {combined_question}"

            codellama_response = run_codellama2(combined_text)
            starcoder2_response = run_starcoder2(combined_text)
            solar_response = run_solar(combined_text)
            mistral_response = run_mistral(combined_text)

            print(f"Prompt: {combined_text}")
            print(f"Response from codellama: {codellama_response}\n")
            print(f"Response from starcoder2: {starcoder2_response}\n")
            print(f"Response from solar: {solar_response}\n")
            print(f"Response from mistral: {mistral_response}\n")

            df.at[index, 'codellama_response'] = codellama_response
            df.at[index, 'starcoder2_response'] = starcoder2_response
            df.at[index, 'solar_response'] = solar_response
            df.at[index, 'mistral_response'] = mistral_response

            time.sleep(2)

        df.to_excel(file_path, index=False)
        print("Excel file has been updated with LLM responses.")

    except Exception as e:
        print(f"Error processing Excel file: {e}")

process_excel_file('./stackexchangeQsAndResponses.xlsx', 5)
