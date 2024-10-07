# import json
# import pandas as pd

# # File paths
# submissions_file_path = 'datascience_submissions'
# comments_file_path = 'datascience_comments'

# # Load comments data first, categorizing by submission ID and selecting the top comment
# comments_by_submission = {}
# with open(comments_file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         try:
#             comment = json.loads(line.strip())
#             # Remove 't3_' prefix
#             link_id = comment['link_id'][3:]
#             if link_id not in comments_by_submission:
#                 comments_by_submission[link_id] = []
#             comments_by_submission[link_id].append({
#                 "body": comment['body'],
#                 "ups": comment.get('ups', 0),
#                 "score": comment.get('score', 0)
#             })
#         except json.JSONDecodeError:
#             continue  # Skip lines that can't be parsed

# # Select the best comment based on 'ups' and 'score'
# def select_best_comment(comments):
#     sorted_comments = sorted(comments, key=lambda x: (-x['ups'], -x['score']))
#     return sorted_comments[0] if sorted_comments else None

# # Collect the best comments
# questions_and_responses = []
# with open(submissions_file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         try:
#             submission = json.loads(line.strip())
#             submission_id = submission['id']
#             if submission_id in comments_by_submission:
#                 best_comment = select_best_comment(comments_by_submission[submission_id])
#                 if best_comment:
#                     # Store only the question and the best response (answer)
#                     questions_and_responses.append({
#                         "question": submission['title'],
#                         "answer": best_comment['body']
#                     })
#                     if len(questions_and_responses) >= 200:
#                         break
#         except json.JSONDecodeError:
#             continue  # Skip lines that can't be parsed

# df = pd.DataFrame(questions_and_responses)

# # Save to Excel file using pandas
# excel_file_path = 'redditQsAndResponses.xlsx'
# df.to_excel(excel_file_path, index=False)

# print(f"Collected {len(questions_and_responses)} questions with the best responses.")
# print(f"Data has been saved to {excel_file_path}.")


# organized_data_file_path = 'organized_questions_with_best_responses-2.json'
# with open(organized_data_file_path, 'w', encoding='utf-8') as file:
#     json.dump(questions_and_responses, file, ensure_ascii=False, indent=4)

# print(f"Collected {len(questions_and_responses)} questions with the best responses.")

import json
import pandas as pd
import re

# File paths
submissions_file_path = '../datascience_submissions'
comments_file_path = '../datascience_comments'

# Function to clean illegal characters from strings for Excel compatibility
def clean_text(text):
    # Replace illegal characters with a space or other placeholder
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)

# Load comments data first, categorizing by submission ID and selecting the top comment
comments_by_submission = {}
with open(comments_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            comment = json.loads(line.strip())
            # Remove 't3_' prefix
            link_id = comment['link_id'][3:]
            if link_id not in comments_by_submission:
                comments_by_submission[link_id] = []
            # Clean the comment body
            comment['body'] = clean_text(comment['body'])
            comments_by_submission[link_id].append({
                "body": comment['body'],
                "ups": comment.get('ups', 0),
                "score": comment.get('score', 0)
            })
        except json.JSONDecodeError:
            continue  # Skip lines that can't be parsed

# Select the best comment based on 'ups' and 'score'
def select_best_comment(comments):
    sorted_comments = sorted(comments, key=lambda x: (-x['ups'], -x['score']))
    return sorted_comments[0] if sorted_comments else None

# Collect the best comments
questions_and_responses = []
with open(submissions_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            submission = json.loads(line.strip())
            submission_id = submission['id']
            if submission_id in comments_by_submission:
                best_comment = select_best_comment(comments_by_submission[submission_id])
                if best_comment:
                    # Clean the submission title and comment body
                    question = clean_text(submission['title'])
                    answer = clean_text(best_comment['body'])
                    # Store only the question and the best response (answer)
                    questions_and_responses.append({
                        "question": question,
                        "answer": answer
                    })
                    # Stop after collecting 25,000 total entries
                    if len(questions_and_responses) >= 25000:
                        break
        except json.JSONDecodeError:
            continue  # Skip lines that can't be parsed

# Split the data
qs_and_answers_test = questions_and_responses[:5000]
qs_and_answers_train = questions_and_responses[5000:25000]

# Convert to DataFrames
df_test = pd.DataFrame(qs_and_answers_test)
df_train = pd.DataFrame(qs_and_answers_train)

# Save to files
test_file_path = 'redditQsAndAnswersTest.xlsx'
train_file_path = 'redditQsAndAnswersTrain.xlsx'

df_test.to_excel(test_file_path, index=False)
df_train.to_excel(train_file_path, index=False)

# Also save to JSON format
with open('redditQsAndAnswersTest.json', 'w', encoding='utf-8') as file:
    json.dump(qs_and_answers_test, file, ensure_ascii=False, indent=4)

with open('redditQsAndAnswersTrain.json', 'w', encoding='utf-8') as file:
    json.dump(qs_and_answers_train, file, ensure_ascii=False, indent=4)

print(f"Collected 5000 questions and responses in the test file and 20,000 in the train file.")

