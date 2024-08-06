from langchain.prompts import ChatPromptTemplate
import os
import getpass
import textwrap
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
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



os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'rag'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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

class RagBot:
    def __init__(self, filepath, model: str=LLM):
        self._client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )
        self._filepath = filepath
        self._model = model
        # Read and split text
        with open(self._filepath, 'r', encoding='utf-8') as file:
            text_content = file.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = text_splitter.create_documents([text_content])
        splits = text_splitter.split_documents(docs)
        # Setup vectorstore
        vector_store = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model=EMBEDDING))
        self._retriever = vector_store.as_retriever()

    @traceable
    def retrieve_docs(self, question):
        return self._retriever.invoke(question)

    # generate 5 diff questions from original question -> retrieve docs relevent to each question 
    # -> get unique union of retrieved docs -> invoke chain on unique union resulting doc
    @traceable
    def get_multiquery(self, original_question: str):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that generates multiple search queries based on a single input query. \n
                    Generate multiple search queries related to: {question} \n
                    Output (4 queries):""",
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

# Continuous interaction with user
rag_bot = RagBot(filepath=content_txt)
while start_session:
    user_input = input("Please enter your question or type 'exit' to quit: ")
    if user_input.lower() == 'exit':
        break
    queries = rag_bot.get_multiquery(user_input)
    documents = rag_bot.retrieve_docs(queries)
    answer = rag_bot.get_answer(user_input)['answer']
    print(answer)

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

first_done=True
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
        experiment_prefix="rag-fusion-qa-oai",
        metadata={"variant": context},
    )
    first_done=True

# Type 2: Answer Hallucination Evaluator (output answer to retrieved docs)
second_done=True
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
        experiment_prefix="rag-fusion-qa-oai-hallucination",
        # Any experiment metadata can be specified here
        metadata={
            "variant": context,
        },
    )
    second_done=True

# Type 3: Document Relevance to Question Evaluator
third_done=True
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
        experiment_prefix="rag-fusion-qa-oai-doc-relevance",
        # Any experiment metadata can be specified here
        metadata={
            "variant": context,
        },
    )
    third_done=True
    quit()

