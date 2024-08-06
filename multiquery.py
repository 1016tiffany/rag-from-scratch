import os
import getpass
import bs4
from langchain.load import dumps, loads
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter



os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'rag'
os.environ['LANGCHAIN_API_KEY'] = getpass.getpass(prompt="Please enter Langchain API key:")
LLM = "mistral:7b"
EMBEDDING = "mxbai-embed-large"
qa_csv = '/Users/petrina/Desktop/Work24/rag-from-scratch/QAset/PsychologyResults.csv'
content_txt = '/Users/petrina/Desktop/Work24/rag-from-scratch/text/CYP2.txt'
dataset_name = 'RAG_multiquery_test_CYP'
context = 'personality psychology, mistral-7b'
description="QA pairs about personality psychology."
topic = 'personality psychology'
start_session = True
is_created = True


# Read file
with open(content_txt, 'r', encoding='utf-8') as file:
    text_content = file.read()

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)
docs = text_splitter.create_documents([text_content])
splits = text_splitter.split_documents(docs)

# Index
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OllamaEmbeddings(model=EMBEDDING))

retriever = vectorstore.as_retriever()
# Multi Query: Different Perspectives

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)
generate_queries = (
    prompt_perspectives 
    | ChatOllama(model=LLM) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
# generate 5 diff questions from original question -> retrieve docs relevent to each question 
# -> get unique union of retrieved docs -> invoke chain on unique union resulting doc
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
# question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = Ollama(model=LLM) 

final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

# print(final_rag_chain.invoke({"question":question}))



# Question
# Continuous interaction with user
while True:
    user_input = input("Please enter your query or type 'exit' to quit: ")
    if user_input.lower() == "exit":
        print("Exiting the program.")
        break
    response = final_rag_chain.invoke(user_input)
    print(response)