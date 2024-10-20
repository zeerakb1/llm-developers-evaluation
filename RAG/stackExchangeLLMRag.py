import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.evaluation import load_evaluator

# Load the StackExchange data from the Excel file
def load_qa_excel(file_path):
    df = pd.read_excel(file_path)
    question_titles = df['Question Title'].tolist()
    question_bodies = df['Question Body'].tolist()
    accepted_answers = df['Accepted Answer Body'].tolist()
    return question_titles, question_bodies, accepted_answers

# Load the query file
def load_query_excel(file_path):
    df = pd.read_excel(file_path)
    query_titles = df['Question Title'].tolist()
    query_bodies = df['Question Body'].tolist()
    queries = [f"Title: {title}\nBody: {body}" for title, body in zip(query_titles, query_bodies)]
    return df, queries

# Create documents
def create_documents(question_titles, question_bodies, accepted_answers):
    documents = []
    for title, body, answer in zip(question_titles, question_bodies, accepted_answers):
        text = f"Question Title: {title}\nQuestion Body: {body}\nAccepted Answer: {answer}"
        documents.append(Document(page_content=text))
    return documents

# Create the FAISS vector store using the embeddings
def create_embeddings(documents):
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, embedder)
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

# Create the RAG chain for different models
def create_rag_chain(model_name, vector_store):
    template = """
    You are a helpful assistant with expertise in coding and technical topics. Use the following context to answer the question as accurately as possible, especially focusing on technical details and code examples if relevant.
    
    Context: {context}
    
    Question: {question}
    
    Provide a helpful and accurate answer, focusing on coding and technical topics.
    """

    prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = OllamaLLM(model=model_name)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt_template}
    )
    return rag_chain

def main(qa_file, query_file, output_file):
    question_titles, question_bodies, accepted_answers = load_qa_excel(qa_file)
    query_df, queries = load_query_excel(query_file)

    documents = create_documents(question_titles, question_bodies, accepted_answers)

    vector_store, embedder = create_embeddings(documents)
    hf_evaluator = load_evaluator("embedding_distance", embeddings=embedder)

    rag_chain_mistral = create_rag_chain("mistral", vector_store)
    rag_chain_codellama = create_rag_chain("codellama", vector_store)
    rag_chain_solar = create_rag_chain("solar", vector_store)
    rag_chain_starcoder2 = create_rag_chain("starcoder2", vector_store)

    mistral_responses = []
    codellama_responses = []
    solar_responses = []
    starcoder2_responses = []

    for query in queries:
        context, distance = get_context_if_relevant(query, vector_store, embedder, hf_evaluator)

        query_context = context if context and distance < 0.5 else ""

        response_mistral = rag_chain_mistral({"query": query, "context": query_context})
        response_codellama = rag_chain_codellama({"query": query, "context": query_context})
        response_solar = rag_chain_solar({"query": query, "context": query_context})
        response_starcoder2 = rag_chain_starcoder2({"query": query, "context": query_context})

        mistral_responses.append(response_mistral['result'])
        codellama_responses.append(response_codellama['result'])
        solar_responses.append(response_solar['result'])
        starcoder2_responses.append(response_starcoder2['result'])

    query_df['mistral_with_rag'] = mistral_responses
    query_df['codellama_with_rag'] = codellama_responses
    query_df['solar_with_rag'] = solar_responses
    query_df['starcoder2_with_rag'] = starcoder2_responses

    query_df.to_excel(output_file, index=False)

    print(f"Number of queries processed: {len(queries)}")

# Main9
if __name__ == "__main__":
    qa_file = '../../stackExchangeQsAndAnswersTest.xlsx'
    query_file = '../Dataset/minDataset/minStackExchangeProcessedResponses.xlsx'
    output_file = 'updated_stackexchange_results.xlsx'
    
    main(qa_file, query_file, output_file)