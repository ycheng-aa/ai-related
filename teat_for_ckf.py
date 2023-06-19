import os
from datetime import datetime
from time import sleep

import backoff
import pandas as pd
from os import listdir

from langchain import OpenAI, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader, TextLoader

from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from logger import logger

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

os.environ["OPENAI_API_KEY"] = "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"

# 公司
# os.environ["OPENAI_API_KEY"] = "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"


os.environ["SERPAPI_API_KEY"] = "de3e259fcac004c1116d823b3cb517fb24df19dc30e0d9486fd9b0644e6582c7"


def generate_prompt():
    prompt_template = """Use the following context to answer the question at the end. If you don't know the answer, just return empty string. {format_instructions}

        {context}

        Question: {question}
        Answer:"""  # noqa

    response_schemas = [
        ResponseSchema(name="college", description="the college of the university which the student is applying to"),
        ResponseSchema(name="major", description="the major which the student is applying to"),
        ResponseSchema(name="second_major", description="the second major which the student is applying to")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"],
        partial_variables={"format_instructions": format_instructions}
    )
    return prompt, output_parser


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def handle_single_pdf():
    loader = TextLoader('/Users/chengyu/stoooges/gpt_test/data/成恺丰.txt', encoding='utf8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vectorstore.as_retriever())
    query = "成恺丰以后的职业规划是什么"
    result = qa.run(query)
    return result



if __name__ == '__main__':
    print(handle_single_pdf())
