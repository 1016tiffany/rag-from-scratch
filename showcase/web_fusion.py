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
from langchain_community.document_loaders import WebBaseLoader

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
qa_csv = '/Users/petrina/Desktop/Work24/rag-from-scratch/QAset/TransformerModel.csv'
dataset_name = 'RAG_InferenceWeb'
context = 'large transformer model inference optimization, mistral-7b'
description="QA pairs about large transformer model inference optimization."
topic = 'large transformer model inference optimization'
start_session = False
is_created = False




class RagBot:
    def __init__(self, model: str = LLM):
        self._client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )
        self._model = model
        # Read and split text
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-01-10-inference-optimization/",),
            bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
                )
            ),
        )
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
    def reciprocal_rank_fusion(self, results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}
        # Iterate through each list of ranked documents
        for documents in results:
        # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(documents):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)
        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        # Return the reranked results as a list of tuples, each containing the document and its fused score
        return reranked_results
    
    @traceable
    def get_answer(self, question: str):
        multiquery = self.get_multiquery(question)
        retrieved_docs = list(map(self.retrieve_docs, multiquery))
        ranked_docs = self.reciprocal_rank_fusion(retrieved_docs)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful AI assistant with expertise in {topic}."
                    " Use the following docs to produce a concise answer to the user question.\n\n"
                    f"## Docs\n\n{ranked_docs}",
                },
                {"role": "user", "content": question},
            ],
        )
# Evaluators will expect "answer" and "contexts"
        return {
            "answer": response.choices[0].message.content,
            "contexts": [str(doc) for doc in ranked_docs],
        }

# question from dataset -> input
# answer from dataset -> reference
# answer from LLM -> prediction

# Go through provided QA csv files and extract question and reference 
df = pd.read_csv(qa_csv)
inputs = df['Question'].tolist()
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
rag_bot = RagBot()

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
        experiment_prefix="rag-fusion-faiss-qa-oai",
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
        experiment_prefix="rag-fusion-faiss-qa-oai-hallucination",
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
        experiment_prefix="rag-fusion-faiss-qa-oai-doc-relevance",
        # Any experiment metadata can be specified here
        metadata={
            "variant": context,
        },
    )
    third_done=True
    quit()

# Continuous interaction with user
while start_session:
    user_input = input("Please enter your query or type 'exit' to quit: ")
    if user_input.lower() == "exit":
        print("Exiting the program.")
        break
    response = rag_bot.get_answer(user_input)['answer']
    print(response)
