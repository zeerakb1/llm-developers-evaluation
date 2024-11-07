# import pandas as pd
# import requests
# import json
# import time

# def get_model_response(model_name, prompt):
#     url = "http://localhost:11434/api/generate"
#     headers = {
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": model_name,
#         "prompt": prompt,
#         "temperature": 0,
#         "stream": False
#     }

#     try:
#         response = requests.post(url, headers=headers, data=json.dumps(data))
#         if response.status_code == 200:
#             try:
#                 response_json = response.json()
#                 return response_json.get("response", "No response available")
#             except json.JSONDecodeError as e:
#                 return f"Failed to decode JSON: {e}"
#         else:
#             return f"Error: {response.status_code}, {response.text}"
#     except Exception as e:
#         return f"An error occurred: {e}"

# def process_excel_file(file_path, num_rows):
#     try:
#         df = pd.read_excel(file_path)
#         df_to_process = df.head(num_rows) 

#         df['codellama_response'] = ""
#         df['mistral_nemo_response'] = ""
#         df['solar_response'] = ""
#         df['mistral_response'] = ""

#         for index, row in df_to_process.iterrows():
#             question_title = row['Question Title']
#             question_body = row['Question Body']
#             combined_question = f"{question_title} {question_body}"
#             combined_text = f"Following is a question posted on a forum, generate a helpful response that is less than 200 words: {combined_question}"

#             codellama_response = get_model_response("codellama:latest", combined_text)
#             mistral_nemo_response = get_model_response("mistral-nemo:12b-instruct-2407-q4_K_M", combined_text)
#             solar_response = get_model_response("solar:latest", combined_text)
#             mistral_response = get_model_response("mistral:latest", combined_text)

#             # Output the responses
#             # print(f"Prompt: {combined_text}")
#             # print(f"Response from codellama: {codellama_response}\n")
#             # print(f"Response from starcoder2: {starcoder2_response}\n")
#             # print(f"Response from solar: {solar_response}\n")
#             # print(f"Response from mistral: {mistral_response}\n")

#             df.at[index, 'codellama_response'] = codellama_response
#             df.at[index, 'mistral_nemo_response'] = mistral_nemo_response
#             df.at[index, 'solar_response'] = solar_response
#             df.at[index, 'mistral_response'] = mistral_response

#             print(f"Row {index + 1} processed.")

#             time.sleep(2)

#         df.to_excel(file_path, index=False)
#         print(f"Excel file has been updated with LLM responses. {num_rows} rows processed.")

#     except Exception as e:
#         print(f"Error processing Excel file: {e}")

# num_rows_to_process = 1025

# process_excel_file('../Dataset/DataExcel/newStackExchangeQsAnswer/stackExchangeQsAndAnswersTest.xlsx', num_rows_to_process)

# import pandas as pd
# import requests
# import json
# import time
# import os
# from openpyxl import load_workbook

# def get_model_response(model_name, prompt):
#     url = "http://localhost:11434/api/generate"
#     headers = {
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": model_name,
#         "prompt": prompt,
#         "temperature": 0,
#         "stream": False
#     }

#     try:
#         response = requests.post(url, headers=headers, data=json.dumps(data))
#         if response.status_code == 200:
#             try:
#                 response_json = response.json()
#                 return response_json.get("response", "No response available")
#             except json.JSONDecodeError as e:
#                 return f"Failed to decode JSON: {e}"
#         else:
#             return f"Error: {response.status_code}, {response.text}"
#     except Exception as e:
#         return f"An error occurred: {e}"

# def process_excel_file(input_file_path, output_file_path, num_rows):
#     try:
#         df = pd.read_excel(input_file_path)
#         df_to_process = df.head(num_rows)

#         # Prepare the new columns
#         columns_to_add = ['codellama_response', 'solar_response', 'mistral_response']

#         # Check if the output file exists; if not, create it with headers
#         if not os.path.exists(output_file_path):
#             pd.DataFrame(columns=df.columns.tolist() + columns_to_add).to_excel(output_file_path, index=False)

#         for index, row in df_to_process.iterrows():
#             question_title = row['Question Title']
#             question_body = row['Question Body']
#             combined_question = f"{question_title} {question_body}"
#             combined_text = f"Following is a question posted on a forum, generate a helpful response that is less than 200 words: {combined_question}"

#             responses = {
#                 "codellama_response": get_model_response("codellama:latest", combined_text),
#                 # "mistral_nemo_response": get_model_response("mistral-nemo:12b-instruct-2407-q4_K_M", combined_text),
#                 "solar_response": get_model_response("solar:latest", combined_text),
#                 "mistral_response": get_model_response("mistral:latest", combined_text)
#             }

#             result_row = pd.DataFrame([{**row.to_dict(), **responses}])

#             with pd.ExcelWriter(output_file_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
#                 result_row.to_excel(writer, index=False, header=writer.sheets == {})

#             print(f"Row {index + 1} processed.")
#             time.sleep(2)

#         print(f"Excel file has been updated with LLM responses. {num_rows} rows processed.")

#     except Exception as e:
#         print(f"Error processing Excel file: {e}")

# num_rows_to_process = 1025

# process_excel_file(
#     '../Dataset/DataExcel/newStackExchangeQsAnswer/stackExchangeQsAndAnswersTest.xlsx',
#     '../Dataset/DataExcel/newStackExchangeQsAnswer/newProcessedFile.xlsx',
#     num_rows_to_process
# )

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

        # Prepare the new columns
        columns_to_add = ['codellama_response', 'solar_response', 'mistral_response']

        # Check if the output file exists; if not, create it with headers
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

            # Append the result to the output file
            with pd.ExcelWriter(output_file_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                result_row.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

            print(f"Row {index + 1} processed.")
            time.sleep(2)

        print(f"Excel file has been updated with LLM responses. {num_rows} rows processed.")

    except Exception as e:
        print(f"Error processing Excel file: {e}")

num_rows_to_process = 525

process_excel_file(
    '../Dataset/DataExcel/newStackExchangeQsAnswer/stackExchangeQsAndAnswersTest.xlsx',
    '../Dataset/DataExcel/newStackExchangeQsAnswer/newProcessedFile.xlsx',
    num_rows_to_process
)
