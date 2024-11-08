import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.evaluation import load_evaluator
from openpyxl import load_workbook
import os

# Import required libraries from HuggingFace for model inference
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datetime import datetime


def append_to_excel(df, output_file):
    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            startrow = writer.sheets['Sheet1'].max_row
            df.to_excel(writer, index=False, header=False, startrow=startrow)
    else:
        df.to_excel(output_file, index=False)

        
# Load the StackExchange data from the Excel file
def load_qa_excel(file_path):
    df = pd.read_excel(file_path)
    question_titles = []
    question_bodies = df['Question Body'].tolist()
    accepted_answers = df['Accepted Answer Body'].tolist()
    return question_titles, question_bodies, accepted_answers

# Load the query file with row limit
def load_query_excel(file_path, n_rows=None):
    df = pd.read_excel(file_path)
    if n_rows is not None:
        df = df.head(n_rows)  # Limit to first n_rows
    query_titles = []
    query_bodies = df['Question Body'].tolist()
    queries = [f"Title: {title}\nBody: {body}" for title, body in zip(query_titles, query_bodies)]
    return df, queries

# Create documents with progress print statements
def create_documents(question_titles, question_bodies, accepted_answers):
    documents = []
    for i, (title, body, answer) in enumerate(zip(question_titles, question_bodies, accepted_answers), 1):
        text = f"Question Body: {body}\nAccepted Answer: {answer}"
        documents.append(Document(page_content=text))
        print(f"Processed document {i}/{len(question_titles)}")
    return documents

# Create the FAISS vector store with a setup progress print statement
def create_embeddings(documents):
    print("Setting up the FAISS vector store with embeddings...")
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, embedder)
    print("FAISS vector store setup completed.")
    return vector_store, embedder

# Retrieval using the embedding distance evaluator
def get_context_if_relevant(query, vector_store, embedder, evaluator):
    retrieved_docs = vector_store.similarity_search(query, k=1)
    if retrieved_docs:
        context = retrieved_docs[0].page_content
        distance_result = evaluator.evaluate_strings(prediction=query, reference=context)
        distance = distance_result['score']
        return context, distance
    return None, 0.0


# Custom model directory and config
output_dir = 'upstage/SOLAR-10.7B-Instruct-v1.0-stackexchange-finetune3'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load custom model and tokenizer
ft_model = AutoModelForCausalLM.from_pretrained(output_dir, quantization_config=bnb_config).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(output_dir)
ft_model.eval()  # Set model to evaluation mode

# Define RAG chain with custom model
def create_rag_chain(vector_store):
    template = """
    You are a helpful assistant with expertise in coding and technical topics. Use the following context to answer the question as accurately as possible, especially focusing on technical details and code examples if relevant.

    Context: {context}

    Question: {question}

    Provide a helpful and accurate answer, focusing on coding and technical topics.
    """
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)
    
    def custom_model_inference(query_context):
        inputs = tokenizer(query_context, return_tensors="pt", truncation=True, max_length=256).to('cuda')
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = ft_model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id)
                answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return answer

    return custom_model_inference

# Main inference loop with custom model
def main(qa_file, query_file, output_file, n_rows=None):
    print(f"Processing {'all' if n_rows is None else n_rows} rows from query file...")
    
    question_titles, question_bodies, accepted_answers = load_qa_excel(qa_file)
    query_df, queries = load_query_excel(query_file, n_rows)

    documents = create_documents(question_titles, question_bodies, accepted_answers)

    vector_store, embedder = create_embeddings(documents)
    hf_evaluator = load_evaluator("embedding_distance", embeddings=embedder)

    rag_chain = create_rag_chain(vector_store)

    for i, query in enumerate(queries, 1):
        context, distance = get_context_if_relevant(query, vector_store, embedder, hf_evaluator)
        query_context = f"Context: {context}\nQuestion: {query}" if context and distance < 0.5 else query

        response = rag_chain(query_context)

        query_df.loc[i - 1, 'custom_model_with_rag'] = response

        append_to_excel(query_df.iloc[[i - 1]], output_file)
        print(f"Processed and saved query {i}/{len(queries)}")

    print(f"Completed processing {len(queries)} queries.")

# Main script
if __name__ == "__main__":
    qa_file = 'Dataset/DataExcel/stackExchangeQsAndAnswersDB.xlsx'
    query_file = 'Dataset/DataExcel/stackExchangeQsAndAnswersTest.xlsx'
    output_file = 'Dataset/rag_and_finetuned_stackexchange_results.xlsx'
    
    n_rows = 1050  # Adjust this value as needed
    main(qa_file, query_file, output_file, n_rows)
