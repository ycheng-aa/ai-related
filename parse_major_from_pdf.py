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
def handle_single_pdf(pdf_file_path, prompt, output_parser):
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=prompt)
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vectorstore.as_retriever())
    query = "What university and major is the student applying to?"
    result = qa.run(query)
    result = output_parser.parse(result)
    del vectorstore
    return result


def main(pdf_directory, start_index=None, end_index=None):
    prompt, output_parser = generate_prompt()
    failed_file_list = []
    df_list = []
    file_list = [i for i in sorted(listdir(pdf_directory)) if i.endswith('.pdf')]
    for idx, file_name in enumerate(file_list):
        try:
            if start_index is not None and idx < start_index:
                continue
            if end_index is not None and idx > end_index:
                continue
            chi_name = file_name.split('_')[0]
            university = file_name.split('_')[1]
            if university.lower() == 'commonapp':
                continue
            file_path = os.path.join(pdf_directory, file_name)
            result = handle_single_pdf(file_path, prompt, output_parser)
            result.update({'chi_name': chi_name, 'file_name': file_name, 'university': university})
            result = pd.Series(result).to_frame().T
            df_list.append(result)
            logger.info(f'file list index: {idx}, parsed {chi_name} doc {file_name}')
            logger.info(f"\n{result}")
            sleep(10)
        except Exception as e:
            logger.exception(f'parse {chi_name} doc {file_name} failed: {str(e)}')
            failed_file_list.append(file_name)
            continue

    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df[['chi_name', 'file_name', 'college', 'major', 'second_major']]
    result_file = f'major_result_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    final_df.to_csv(result_file)
    logger.info(f'failed file list are: {failed_file_list}')


if __name__ == '__main__':
    main('/Users/chengyu/Downloads/tmp_preview_293/', 0, 2)
