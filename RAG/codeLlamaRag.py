import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Load the question-answer data from the Excel file
def load_qa_excel(file_path):
    df = pd.read_excel(file_path)
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    return questions, answers

# Load the query file 
def load_query_excel(file_path):
    df = pd.read_excel(file_path)
    queries = df['question'].tolist()
    return df, queries

# Create a list of documents with question-answer pairs
def create_documents(questions, answers):
    documents = []
    for question, answer in zip(questions, answers):
        text = f"Question: {question}\nAnswer: {answer}"
        documents.append(Document(page_content=text))
    return documents

# Create the FAISS vector store using the embeddings
def create_embeddings(documents):
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, embedder)
    return vector_store

# Create the RAG chain using a custom prompt template and OllamaLLM (with CodeLlama model)
def create_rag_chain(vector_store):
    template = """
    You are a helpful assistant knowledgeable in programming and technical topics. Use the following context to answer the question as accurately as possible.
    
    Context: {context}
    
    Question: {question}
    
    Provide a helpful and accurate answer, especially related to code and technical details.
    """

    prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = Ollama(model="codellama")

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt_template}
    )
    return rag_chain

def main(qa_file, query_file, output_file):
    questions, answers = load_qa_excel(qa_file)
    query_df, queries = load_query_excel(query_file)
    documents = create_documents(questions, answers)

    vector_store = create_embeddings(documents)

    rag_chain = create_rag_chain(vector_store)

    responses = []

    for query in queries:
        response = rag_chain({"query": query})
        responses.append(response['result'])

    query_df['codellama_with_rag'] = responses

    query_df.to_excel(output_file, index=False)

    print(f"Number of queries processed: {len(queries)}")

# Main
if __name__ == "__main__":
    qa_file = '../redditQsAndAnswersTest.xlsx'
    query_file = '../minResponsesProcessedReddit.xlsx'
    output_file = 'updated_query_results.xlsx'
    
    main(qa_file, query_file, output_file)
