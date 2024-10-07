from lxml import etree
import pandas as pd
import json

# Function to extract data from XML with limits and starting index
def extract_data_from_xml(xml_file, limit=None, start=0):
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(xml_file, parser=parser)
    root = tree.getroot()

    data = []
    count = 0
    skip_count = start

    for row in root.findall('.//row'):
        if limit and count >= limit:
            break  # Stop extracting if the limit is reached

        accepted_answer_id = row.get('AcceptedAnswerId')

        if skip_count > 0:
            skip_count -= 1
            continue

        if accepted_answer_id:
            question_id = row.get('Id')
            question_title = row.get('Title')
            question_body = row.get('Body')

            accepted_answer_row = root.find(f'.//row[@Id="{accepted_answer_id}"]')

            if accepted_answer_row is not None:
                accepted_answer_body = accepted_answer_row.get('Body')
                data.append({
                    'Question Title': question_title,
                    'Question Body': question_body,
                    'Accepted Answer Body': accepted_answer_body
                })
                count += 1

    return data

def extract_data_from_multiple_files(xml_file1, xml_file2, limit, start=0):
    data = extract_data_from_xml(xml_file1, limit=limit, start=start)

    if len(data) < limit:
        remaining_limit = limit - len(data)
        data_from_second_file = extract_data_from_xml(xml_file2, limit=remaining_limit)
        data.extend(data_from_second_file)

    return data

def save_to_excel(data, output_file):
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)

def save_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

input_xml_file1 = './datascience.stackexchange.com/Posts.xml'
input_xml_file2 = './datascience.stackexchange2.com/Posts.xml'

data_test = extract_data_from_multiple_files(input_xml_file1, input_xml_file2, limit=5000)

data_train = extract_data_from_multiple_files(input_xml_file1, input_xml_file2, limit=20000, start=5000)

print(f"Total Test Set: {len(data_test)}")
print(f"Total Train Set: {len(data_train)}")

output_test_excel = 'stackExchangeQsAndAnswersTest.xlsx'
output_test_json = 'stackExchangeQsAndAnswersTest.json'
save_to_excel(data_test, output_test_excel)
save_to_json(data_test, output_test_json)

output_train_excel = 'stackExchangeQsAndAnswersTrain.xlsx'
output_train_json = 'stackExchangeQsAndAnswersTrain.json'
save_to_excel(data_train, output_train_excel)
save_to_json(data_train, output_train_json)

print(f"Extracted and saved 5000 questions and answers to {output_test_excel} and {output_test_json}.")
print(f"Extracted and saved 20,000 questions and answers to {output_train_excel} and {output_train_json}.")