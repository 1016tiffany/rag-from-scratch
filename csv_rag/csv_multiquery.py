import os
import getpass
import time
import bs4
import tiktoken
import pandas as pd
from langsmith import Client
from langsmith import traceable
from langchain_ollama import ChatOllama
from langchain.load import dumps, loads
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
os.environ['LANGCHAIN_PROJECT'] = 'rag'
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass(prompt='Please enter Langchain API key:')
LLM = "mistral:7b"
EMBEDDING = "mxbai-embed-large"
qa_csv = '/Users/petrina/Desktop/Work24/rag-from-scratch/QAset/OlympicsMedals.csv'
content_csv = '/Users/petrina/Desktop/Work24/rag-from-scratch/csv/olympics_medals_country_wise.csv'
dataset_name = 'RAG_test_OlympicsMedalbyCountry'
context = 'Olympics medal by country, mistral-7b'
description="QA pairs about Olympics medal."
topic = 'Olympics and mathematics'
start_session = False
is_created = True


class RagBot:
    def __init__(self, filepath, model: str = LLM):
        self._client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )
        self._filepath = filepath
        self._model = model
        # Read and split text
        loader = CSVLoader(file_path=self._filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        # Setup vectorstore
        vector_store = FAISS.from_documents(documents=splits, embedding=OllamaEmbeddings(model=EMBEDDING))
        self._retriever = vector_store.as_retriever()

    @traceable()
    def retrieve_docs(self, question):
        return self._retriever.invoke(question)
    
    @traceable
    def get_multiquery(self, original_question: str):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI language model assistant. Your task is to generate five 
                    different versions of the given user question to retrieve relevant documents from a vector 
                    database. By generating multiple perspectives on the user question, your goal is to help
                    the user overcome some of the limitations of the distance-based similarity search. 
                    Provide these alternative questions separated by newlines. Original question: {original_question}""",
                },
                {"role": "user", "content": original_question},
            ],
        )
        return {
            "answer": response.choices[0].message.content.split('\n')
        }
    
    @traceable
    def get_unique_union(self, documents: list[list]):
        """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
        unique_docs = list(set(flattened_docs))
    # Return
        return [loads(doc) for doc in unique_docs]
    
    @traceable
    def get_answer(self, question: str):
        multiquery = self.get_multiquery(question)
        retrieved_docs = list(map(self.retrieve_docs, multiquery))
        unique_docs = self.get_unique_union(retrieved_docs)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful AI assistant with expertise in {topic}."
                    " Use the following docs to produce a concise answer to the user question.\n\n"
                    f"## Docs\n\n{unique_docs}",
                },
                {"role": "user", "content": question},
            ],
        )
# Evaluators will expect "answer" and "contexts"
        return {
            "answer": response.choices[0].message.content,
            "contexts": [str(doc) for doc in unique_docs],
        }
    


# question from dataset -> input
# answer from dataset -> reference
# answer from LLM -> prediction

# Go through provided QA csv files and extract question and reference 
df = pd.read_csv(qa_csv)
inputs = df['Question Input'].tolist()
outputs = df['Correct Answer'].tolist()
qa_pairs = [{"question": q, "answer": a} for q, a in zip(inputs, outputs)]
client = Client()

# Create dataset
if not is_created:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )
    client.create_examples(
        inputs=[{"question": q} for q in inputs],
        outputs=[{"answer": a} for a in outputs],
        dataset_id=dataset.id,
    )
    is_created = True

# RAG chain
def predict_rag_answer(example: dict):
    """Use this for answer evaluation"""
    response = rag_bot.get_answer(example["question"])
    return {"answer": response["answer"]}

def predict_rag_answer_with_context(example: dict):
    """Use this for evaluation of retrieved documents and hallucinations"""
    response = rag_bot.get_answer(example["question"])
    return {"answer": response["answer"], "contexts": response["contexts"]}

# Evaluator LLM
eval_llm = ChatOllama(model=LLM)
rag_bot = RagBot(filepath=content_csv)

first_done=False
# Type 1: Reference Answer Evaluator (output answer to reference answer)
qa_evalulator = [
    LangChainStringEvaluator(
        # contextual accuracy
        "cot_qa",
        config={"llm": eval_llm},
        prepare_data=
        lambda run, example: {
            "prediction": run.outputs["answer"],
            "reference": example.outputs["answer"],
            "input": example.inputs["question"],
        },
    )
]
if not first_done:
    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=qa_evalulator,
        experiment_prefix="rag-multiquery-faiss-qa-oai",
        metadata={"variant": context},
    )
    first_done=True

# Type 2: Answer Hallucination Evaluator (output answer to retrieved docs)
second_done=False
answer_hallucination_evaluator = LangChainStringEvaluator(
    "labeled_score_string",
    config={
        "llm": eval_llm,
        "criteria": {
            "accuracy": """Is the Assistant's Answer grounded in the Ground Truth documentation? A score of [[1]] means that the
            Assistant answer contains is not at all based upon / grounded in the Groun Truth documentation. A score of [[5]] means 
            that the Assistant answer contains some information (e.g., a hallucination) that is not captured in the Ground Truth 
            documentation. A score of [[10]] means that the Assistant answer is fully based upon the in the Ground Truth documentation."""
        },
        # If you want the score to be saved on a scale from 0 to 1
        "normalize_by": 10,
    },
    prepare_data=
    lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": run.outputs["contexts"],
        "input": example.inputs["question"],
    },
)

if not second_done:
    experiment_results = evaluate(
        predict_rag_answer_with_context,
        data=dataset_name,
        evaluators=[answer_hallucination_evaluator],
        experiment_prefix="rag-multiquery-faiss-qa-oai-hallucination",
        # Any experiment metadata can be specified here
        metadata={
            "variant": context,
        },
    )
    second_done=True

# Type 3: Document Relevance to Question Evaluator
third_done=False
docs_relevance_evaluator = LangChainStringEvaluator(
    "score_string",
    config={
        "llm": eval_llm,
        "criteria": {
            "document_relevance": textwrap.dedent(
                """The response is a set of documents retrieved from a vectorstore. The input is a question
            used for retrieval. You will score whether the Assistant's response (retrieved docs) is relevant to the Ground Truth 
            question. A score of [[1]] means that none of the  Assistant's response documents contain information useful in answering or addressing the user's input.
            A score of [[5]] means that the Assistant answer contains some relevant documents that can at least partially answer the user's question or input. 
            A score of [[10]] means that the user input can be fully answered using the content in the first retrieved doc(s)."""
            )
        },
        # If you want the score to be saved on a scale from 0 to 1
        "normalize_by": 10,
    },
    prepare_data=
    lambda run, example: {
        "prediction": run.outputs["contexts"],
        "input": example.inputs["question"],
    },
)

if not third_done:
    experiment_results = evaluate(
        predict_rag_answer_with_context,
        data=dataset_name,
        evaluators=[docs_relevance_evaluator],
        experiment_prefix="rag-multiquery-faiss-qa-oai-doc-relevance",
        # Any experiment metadata can be specified here
        metadata={
            "variant": context,
        },
    )
    third_done=True
    quit()

# Continuous interaction with user

rag_bot = RagBot(filepath=content_csv)
while start_session:
    user_input = input("Please enter your query or type 'exit' to quit: ")
    if user_input.lower() == "exit":
        print("Exiting the program.")
        break
    response = rag_bot.get_answer(user_input)['answer']
    print(response)

# def stt():
#     mic = WhisperMic(mic_index=0)
#     result = mic.listen()
#     print("Transcription: ", result)
#     print(type(result))
#     return result

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
