import pandas as pd
from sentence_transformers import SentenceTransformer, util
from bert_score import BERTScorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import evaluate
import nltk
import json

nltk.download('punkt')

bertscore = evaluate.load("bertscore")

def cosine_similarity_score(sentences1, sentences2):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    # return cos_scores.numpy().diagonal()
    return cos_scores.cpu().numpy().diagonal()

# Function to compute BERTScore
def compute_bert_score(sentences1, sentences2):
    results = bertscore.compute(predictions=sentences1, references=sentences2, model_type='bert-base-uncased', lang="en")
    return results['f1']

def compute_meteor_score(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)
    return meteor_score([reference_tokens], candidate_tokens)

def compute_bleu_score(reference, candidate):
    reference_tokens = [nltk.word_tokenize(reference)]
    candidate_tokens = nltk.word_tokenize(candidate)
    bleu = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu

def compare_responses(df):
    scores_list = []

    model_columns = ['codellama_response', 'starcoder2_response', 'solar_response', 'mistral_response']
    answer_column = 'answer'
    
    for model in model_columns:
        col1 = df[answer_column].tolist()
        col2 = df[model].tolist()

        bert_scores = compute_bert_score(col2, col1)  # BERTScore
        meteor_scores = [compute_meteor_score(col1[i], col2[i]) for i in range(len(col1))]  # METEOR
        bleu_scores = [compute_bleu_score(col1[i], col2[i]) for i in range(len(col1))]  # BLEU-4
        cosine_scores = cosine_similarity_score(col1, col2)  # Cosine Similarity

        avg_bert = float(sum(bert_scores) / len(bert_scores))
        avg_meteor = float(sum(meteor_scores) / len(meteor_scores))
        avg_bleu = float(sum(bleu_scores) / len(bleu_scores))
        avg_cosine_similarity = float(sum(cosine_scores) / len(cosine_scores))

        print(f"Model: {model}, Avg BERT: {avg_bert}, Avg METEOR: {avg_meteor}, Avg BLEU-4: {avg_bleu}, Avg Cosine: {avg_cosine_similarity}")

        scores_list.append({
            'Model Type': model,
            'BERT Score': avg_bert,
            'METEOR Score': avg_meteor,
            'BLEU-4 Score': avg_bleu,
            'Cosine Similarity': avg_cosine_similarity
        })

    return scores_list

file_path = 'redditQsAndResponses.xlsx'
df = pd.read_excel(file_path)

for col in df.columns:
    df[col] = df[col].astype(str)

scores = compare_responses(df)

json_file_path = 'evaluation_results_reddit.json'
with open(json_file_path, 'w') as json_file:
    json.dump(scores, json_file, indent=4)

print(f"Results saved to {json_file_path}")
