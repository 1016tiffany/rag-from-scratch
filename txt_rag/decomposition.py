from langchain.prompts import ChatPromptTemplate
import os
import getpass
import textwrap
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Message

from langsmith import traceable
from langsmith import Client
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS


from langchain.prompts import ChatPromptTemplate
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'rag'
os.environ['LANGCHAIN_API_KEY'] = getpass.getpass(prompt="Please enter Langchain API key:")
LLM = "mistral:7b"
EMBEDDING = "mxbai-embed-large"
content_csv = '/Users/petrina/Desktop/Work24/rag-from-scratch/csv/olympics_medals_country_wise.csv'
dataset_name = 'RAG_test_OlympicsMedalbyCountry'
context = 'Olympics medal by country, mistral-7b'
description="QA pairs about Olympics medal."
topic = 'Olympics and mathematics'
start_session = False
is_created = True

class RagBot:
    def __init__(self, filepath, model: str=LLM):
        # Wrapping the client instruments the LLM
        # self._client = Ollama.Client()
        self.client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )
        self.filepath = filepath
        self.model = model
        # Read and split text
        with open(self._filepath, 'r', encoding='utf-8') as file:
            text_content = file.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = text_splitter.create_documents([text_content])
        splits = text_splitter.split_documents(docs)
        # Setup vectorstore
        vector_store = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model=EMBEDDING))
        self.retriever = vector_store.as_retriever()
        
    @traceable
    def retrieve_docs(self, question):
        return self.retriever.invoke(question)

    # generate 5 diff questions from original question -> retrieve docs relevent to each question 
    # -> get unique union of retrieved docs -> invoke chain on unique union resulting doc
    @traceable
    def get_decomposition(self, question: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
                    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
                    Generate multiple search queries related to: {question} \n
                    Output (3 queries):""",
                },
                {"role": "user", "content": question},
            ],
        )
        return {
            "answer": response.choices[0].message.content.split('\n')
        }
    
    @traceable
    def get_answers(self, questions):
        q_a_pairs = []
        for question in questions:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": question
                    }(role="user", content=question)]
            )
            q_a_pairs.append((question, response.choices[0].message.content))
        return q_a_pairs
    
    @traceable
    def format_qa_pairs(self, q_a_pairs):
        return '\n'.join([f"Question: {q}\nAnswer: {a}\n---" for q, a in q_a_pairs])
    
    @traceable
    def question_with_context(self, question: str):
        sub_questions = self.get_decomposition(question)
        qa_pairs = self.get_answers(sub_questions)
        formatted_qa = self.format_qa_pairs(qa_pairs)
        return formatted_qa

    @traceable
    def answer_question_with_context(self, question: str, context: str):
        context = self.question_with_context
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Here is the question you need to answer:
                    \n --- \n {question} \n --- \n
                    Here is any available background question + answer pairs:
                    \n --- \n {q_a_pairs} \n --- \n
                    Here is additional context relevant to the question: 
                    \n --- \n {context} \n --- \n
                    Use the above context and any background question + answer pairs to answer the question: \n {question}""",
                },
                {"role": "user", "content": question},
            ],
        )
# Evaluators will expect "answer" and "contexts"
        return {
            "answer": response.choices[0].message.content,
            "contexts": [str(doc) for doc in context],
        }

rag_bot = RagBot(filepath=content_csv)
while start_session:
    user_input = input("Please enter your query or type 'exit' to quit: ")
    if user_input.lower() == "exit":
        print("Exiting the program.")
        break
    response = rag_bot.get_answer(user_input)['answer']
    print(response)
    


# # Decomposition
# template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
# The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
# Generate multiple search queries related to: {question} \n
# Output (3 queries):"""
# prompt_decomposition = ChatPromptTemplate.from_template(template)

# # LLM
# llm = Ollama(model=LLM) 

# # Read file
# with open(content_txt, 'r', encoding='utf-8') as file:
#     text_content = file.read()

# # Split
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=300, 
#     chunk_overlap=50)
# docs = text_splitter.create_documents([text_content])
# splits = text_splitter.split_documents(docs)

# # Index
# vectorstore = Chroma.from_documents(documents=splits, 
#                                     embedding=OllamaEmbeddings(model=EMBEDDING))

# retriever = vectorstore.as_retriever()

# # Chain
# generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

# # Run
# question = "What are the main components of an LLM-powered autonomous agent system?"
# questions = generate_queries_decomposition.invoke({"question":question})
# print(questions)

# # ANSWER RECUSIVELY
# # Prompt
# template = """Here is the question you need to answer:

# \n --- \n {question} \n --- \n

# Here is any available background question + answer pairs:

# \n --- \n {q_a_pairs} \n --- \n

# Here is additional context relevant to the question: 

# \n --- \n {context} \n --- \n

# Use the above context and any background question + answer pairs to answer the question: \n {question}
# """

# decomposition_prompt = ChatPromptTemplate.from_template(template)

# def format_qa_pair(question, answer):
#     """Format Q and A pair"""
    
#     formatted_string = ""
#     formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
#     return formatted_string.strip()

# # llm
# llm = Ollama(model=LLM) 

# q_a_pairs = ""
# for q in questions:
    
#     rag_chain = (
#     {"context": itemgetter("question") | retriever, 
#      "question": itemgetter("question"),
#      "q_a_pairs": itemgetter("q_a_pairs")} 
#     | decomposition_prompt
#     | llm
#     | StrOutputParser())

#     answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
#     q_a_pair = format_qa_pair(q,answer)
#     q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
# print(answer)

# # ANSWER INDIVIDUALLY
