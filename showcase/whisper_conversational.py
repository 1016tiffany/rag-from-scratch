import os
import getpass
import time
import bs4
import tiktoken
import pandas as pd
from langsmith import Client
from langsmith import traceable
from langchain_ollama import ChatOllama
from openai import OpenAI
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
import textwrap
from langchain_community.embeddings import OllamaEmbeddings
import ollama
from langchain_community.vectorstores import FAISS

# speech recognition
import whisper
from whisper_mic import WhisperMic
os.environ['KMP_DUPLICATE_LIB_OK']='True'

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'rag_or_llm'
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass(prompt='Please enter Langchain API key:')

LLM = "mistral:7b"
EMBEDDING = "mxbai-embed-large"
qa_csv = '/Users/petrina/Desktop/Work24/rag-from-scratch/QAset/OlympicsMedals.csv'
content_csv = '/Users/petrina/Desktop/Work24/rag-from-scratch/csv/olympics_medals_country_wise.csv'
dataset_name = 'RAG_test_OlympicsMedalbyCountry'
context = 'Olympics medal by country, mistral-7b'
description="QA pairs about Olympics medal."
topic = 'Olympics and mathematics'
start_session = True
is_created = True
RELEVANCE_THRESHOLD = 0.7

class RagBot:
    def __init__(self, filepath, model: str = LLM, embedding=EMBEDDING):
        self.client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )
        self.filepath = filepath
        self.model = model
        # Read and split text
        loader = CSVLoader(file_path=self.filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        # Setup vectorstore
        # Initialize vector store with FAISS

        # self._index, self._docs = self.create_faiss_index(splits, embedding)
        self.embeddings = OllamaEmbeddings(model=EMBEDDING)
        self.db = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.db.as_retriever()


        # vector_store = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model=EMBEDDING))
        # self._retriever = vector_store.as_retriever()

    
    def retrieve_docs_with_scores(self, question, top_k=3):
        pass
        # docs_and_scores = self.db.similarity_search_with_score(question)
        # return docs_and_scores[0]


    @traceable()
    def retrieve_docs(self, question):
        return self.retriever.invoke(question)
    
    @traceable
    def rag_or_llm(self, question):
        docs_and_scores = self.db.similarity_search_with_score(question)
        retrieved_docs, scores = docs_and_scores[0]
        
        # retrieved_docs, scores = zip(*self.retrieve_docs_with_scores(question))
        if scores > RELEVANCE_THRESHOLD:
            formatted_docs = "\n\n".join([str(doc) for doc in retrieved_docs])
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful AI assistant with expertise in specific topics."
                                   " Use the following docs to produce a concise answer to the user question.\n\n"
                                   f"## Docs\n\n{formatted_docs}",
                    },
                    {"role": "user", "content": question},
                ],
            ).choices[0].message.content
        else:
            return self.client.chat.completions.create(
                model=ChatOllama(model=LLM),
                messages=[{"role": "user", "content": question}],
            ).choices[0].message.content

@traceable
def stt():
    mic = WhisperMic(mic_index=0)
    result = mic.listen()
    print("Transcription: ", result)
    return result

# @traceable
# def main():
#     result_txt = stt()
#     return result_txt
#     # result_txt = "Can you give me some water?"


# if __name__ == "__main__":
#     start_time = time.time()
#     main()
#     end_time = time.time()
#     total_time = end_time - start_time 
#     print(f"latency: {total_time:.4f} seconds")


def main():
    while start_session:
        user_input = stt()
        normalized_input = user_input.strip().lower()
        print("Normalized Input: [{}]".format(normalized_input))
        if normalized_input == "i want to exit the program.":
            print("Exiting the program.")
            break
        response = rag_bot.rag_or_llm(user_input)
        print(response)

if __name__ == "__main__":
    rag_bot = RagBot(filepath=content_csv)
    main()



