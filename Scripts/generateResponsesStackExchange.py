import pandas as pd
import requests
import json
import time
import os
from openpyxl import load_workbook

def get_model_response(model_name, prompt):
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            try:
                response_json = response.json()
                return response_json.get("response", "No response available")
            except json.JSONDecodeError as e:
                return f"Failed to decode JSON: {e}"
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"An error occurred: {e}"

def process_excel_file(input_file_path, output_file_path, num_rows):
    try:
        df = pd.read_excel(input_file_path)
        df_to_process = df.head(num_rows)

        columns_to_add = ['codellama_response', 'solar_response', 'mistral_response']

        if not os.path.exists(output_file_path):
            pd.DataFrame(columns=df.columns.tolist() + columns_to_add).to_excel(output_file_path, index=False)

        for index, row in df_to_process.iterrows():
            question_title = row['Question Title']
            question_body = row['Question Body']
            combined_question = f"{question_title} {question_body}"
            combined_text = f"Following is a question posted on a forum, generate a helpful response that is less than 200 words: {combined_question}"

            responses = {
                "codellama_response": get_model_response("codellama:latest", combined_text),
                "solar_response": get_model_response("solar:latest", combined_text),
                "mistral_response": get_model_response("mistral:latest", combined_text)
            }

            result_row = pd.DataFrame([{**row.to_dict(), **responses}])

            with pd.ExcelWriter(output_file_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                result_row.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

            print(f"Row {index + 1} processed.")
            time.sleep(2)

        print(f"Excel file has been updated with LLM responses. {num_rows} rows processed.")

    except Exception as e:
        print(f"Error processing Excel file: {e}")

num_rows_to_process = 1050

process_excel_file(
    '../Dataset/DataExcel/newStackExchangeQsAnswer/stackExchangeQsAndAnswersTest.xlsx',
    '../Dataset/DataExcel/newStackExchangeQsAnswer/newProcessedFile.xlsx',
    num_rows_to_process
)
