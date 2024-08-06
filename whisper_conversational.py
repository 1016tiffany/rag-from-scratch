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

# question from dataset -> input
# answer from dataset -> reference
# answer from LLM -> prediction

# Go through provided QA csv files and extract question and reference 
# df = pd.read_csv(qa_csv)
# inputs = df['Question Input'].tolist()
# outputs = df['Correct Answer'].tolist()
# qa_pairs = [{"question": q, "answer": a} for q, a in zip(inputs, outputs)]
# client = Client()

# # Create dataset
# if not is_created:
#     dataset = client.create_dataset(
#         dataset_name=dataset_name,
#         description=description,
#     )
#     client.create_examples(
#         inputs=[{"question": q} for q in inputs],
#         outputs=[{"answer": a} for a in outputs],
#         dataset_id=dataset.id,
#     )
#     is_created = True

# # RAG chain
# def predict_rag_answer(example: dict):
#     """Use this for answer evaluation"""
#     response = rag_bot.get_answer(example["question"])
#     return {"answer": response["answer"]}

# def predict_rag_answer_with_context(example: dict):
#     """Use this for evaluation of retrieved documents and hallucinations"""
#     response = rag_bot.get_answer(example["question"])
#     return {"answer": response["answer"], "contexts": response["contexts"]}

# # Evaluator LLM
# eval_llm = ChatOllama(model=LLM)
# rag_bot = RagBot(filepath=content_csv)

# first_done=False
# # Type 1: Reference Answer Evaluator (output answer to reference answer)
# qa_evalulator = [
#     LangChainStringEvaluator(
#         # contextual accuracy
#         "cot_qa",
#         config={"llm": eval_llm},
#         prepare_data=
#         lambda run, example: {
#             "prediction": run.outputs["answer"],
#             "reference": example.outputs["answer"],
#             "input": example.inputs["question"],
#         },
#     )
# ]
# if not first_done:
#     experiment_results = evaluate(
#         predict_rag_answer,
#         data=dataset_name,
#         evaluators=qa_evalulator,
#         experiment_prefix="rag-basic-qa-oai",
#         metadata={"variant": context},
#     )
#     first_done=True

# # Type 2: Answer Hallucination Evaluator (output answer to retrieved docs)
# second_done=False
# answer_hallucination_evaluator = LangChainStringEvaluator(
#     "labeled_score_string",
#     config={
#         "llm": eval_llm,
#         "criteria": {
#             "accuracy": """Is the Assistant's Answer grounded in the Ground Truth documentation? A score of [[1]] means that the
#             Assistant answer contains is not at all based upon / grounded in the Groun Truth documentation. A score of [[5]] means 
#             that the Assistant answer contains some information (e.g., a hallucination) that is not captured in the Ground Truth 
#             documentation. A score of [[10]] means that the Assistant answer is fully based upon the in the Ground Truth documentation."""
#         },
#         # If you want the score to be saved on a scale from 0 to 1
#         "normalize_by": 10,
#     },
#     prepare_data=
#     lambda run, example: {
#         "prediction": run.outputs["answer"],
#         "reference": run.outputs["contexts"],
#         "input": example.inputs["question"],
#     },
# )

# if not second_done:
#     experiment_results = evaluate(
#         predict_rag_answer_with_context,
#         data=dataset_name,
#         evaluators=[answer_hallucination_evaluator],
#         experiment_prefix="rag-basic-qa-oai-hallucination",
#         # Any experiment metadata can be specified here
#         metadata={
#             "variant": context,
#         },
#     )
#     second_done=True

# # Type 3: Document Relevance to Question Evaluator
# third_done=False
# docs_relevance_evaluator = LangChainStringEvaluator(
#     "score_string",
#     config={
#         "llm": eval_llm,
#         "criteria": {
#             "document_relevance": textwrap.dedent(
#                 """The response is a set of documents retrieved from a vectorstore. The input is a question
#             used for retrieval. You will score whether the Assistant's response (retrieved docs) is relevant to the Ground Truth 
#             question. A score of [[1]] means that none of the  Assistant's response documents contain information useful in answering or addressing the user's input.
#             A score of [[5]] means that the Assistant answer contains some relevant documents that can at least partially answer the user's question or input. 
#             A score of [[10]] means that the user input can be fully answered using the content in the first retrieved doc(s)."""
#             )
#         },
#         # If you want the score to be saved on a scale from 0 to 1
#         "normalize_by": 10,
#     },
#     prepare_data=
#     lambda run, example: {
#         "prediction": run.outputs["contexts"],
#         "input": example.inputs["question"],
#     },
# )

# if not third_done:
#     experiment_results = evaluate(
#         predict_rag_answer_with_context,
#         data=dataset_name,
#         evaluators=[docs_relevance_evaluator],
#         experiment_prefix="rag-basic-qa-oai-doc-relevance",
#         # Any experiment metadata can be specified here
#         metadata={
#             "variant": context,
#         },
#     )
#     third_done=True
#     quit()

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

# # Continuous interaction with user
# rag_bot = RagBot(filepath=content_csv)
# while start_session:
#     user_input = stt()
#     print(type(user_input))
#     if user_input == "I want to exit the program.":
#         print("Exiting the program.")
#         break
#     print('fail')
#     response = rag_bot.rag_or_llm(user_input)
#     print(response)

