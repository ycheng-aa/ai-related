import os
import pandas as pd
import re
from datetime import datetime
from time import sleep

import pandas as pd

from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, QuestionAnswerPrompt

from logger import logger

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

os.environ["OPENAI_API_KEY"] = "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"


def main():
    pdf_directory = '/Users/chengyu/Downloads/tmp_preview_293/'
    file_list = ['万熙培_University of Pennsylvania_2022-23_1.pdf', '万熙培_Washington University in St.Louis_2022-23_1.pdf',
                 '严一苇_Cornell University_2021-22_1.pdf', '严一苇_Cornell University_2021-22_2.pdf',
                 '严嘉仪_Cornell University_2021-22_2.pdf', '严嘉仪_Duke University_2021-22_2.pdf',
                 '严嘉仪_Johns Hopkins University_2021-22_2.pdf', '严嘉仪_University of Southern California_2021-22_2.pdf',
                 '严嘉仪_Washington University in St.Louis_2021-22_2.pdf', '于婧蕾_Boston College_2021-22_2.pdf',
                 '于宗琪_Brown University_2021-22_transfer_1.pdf', '于宗琪_Columbia University_2022-23_transfer_1.pdf',
                 '于宗琪_Cornell University_2021-22_transfer_1.pdf', '于宗琪_Duke University_2021-22_transfer_1.pdf',
                 '于宗琪_Harvard University_2021-22_transfer_1.pdf', '于宗琪_Northwestern University_2021-22_transfer_1.pdf',
                 '于宗琪_Stanford University_2021-22_transfer_1.pdf', '于宗琪_University of Chicago_2022-23_transfer_1.pdf',
                 '于宗琪_University of Pennsylvania_2021-22_transfer_1.pdf', '于宗琪_Yale University_2021-22_transfer_1.pdf',
                 '于馨淼_Barnard College_2022-23_2.pdf', '于馨淼_Carnegie Mellon University_2022-23_2.pdf',
                 '于馨淼_Cornell University_2022-23_1.pdf', '于馨淼_Duke University_2022-23_1.pdf',
                 '于馨淼_Rice University_2022-23_2.pdf', '于馨淼_University of Southern California_2022-23_1.pdf',
                 '付晨溪_Rice University_2022-23_2.pdf', '付晨溪_University of Southern California_2022-23_2.pdf',
                 '付晨溪_Washington University in St.Louis_2022-23_2.pdf', '付毅成_Cornell University_2022-23_transfer_1.pdf',
                 '付毅成_University of Michigan-Ann Arbor_2022-23_transfer_1.pdf',
                 '付毅成_Washington University in St.Louis_2022-23_transfer_1.pdf',
                 '任越麒_Carnegie Mellon University_2022-23_2.pdf', '任越麒_University of Rochester_2022-23_2.pdf',
                 '任越麒_University of Southern California_2022-23_2.pdf',
                 '任越麒_Washington University in St.Louis_2022-23_2.pdf', '何嘉乐_Northeastern University_2022-23_1.pdf',
                 '何嘉乐_Pennsylvania State University_2022-23_1.pdf', '何山_New York University_2022-23_1.pdf',
                 '何山_Ohio State University_2022-23_1.pdf', '何山_UC_2022-23.pdf', '余懿炜_Brown University_2021-22_2.pdf',
                 '余懿炜_Carnegie Mellon University_2021-22_1.pdf', '余懿炜_Cornell University_2021-22_1.pdf',
                 '余懿炜_Duke University_2021-22_2.pdf', '余懿炜_University of Pennsylvania_2021-22_2.pdf',
                 '余懿炜_Washington University in St.Louis_2021-22_1.pdf', '余懿炜_Wellesley College_2021-22_2.pdf',
                 '侯妙严_Amherst College_2022-23_2.pdf', '侯妙严_Duke University_2022-23_1.pdf',
                 '侯妙严_Harvard University_2022-23_1.pdf', '侯妙严_Smith College_2022-23_1.pdf',
                 '侯笑恬_Barnard College_2022-23_2.pdf', '侯笑恬_Brown University_2022-23_2.pdf',
                 '侯笑恬_Cornell University_2022-23_2.pdf', '俞泓辰_McGill University_2021-22_1.pdf',
                 '倪泽彬_Bates College_2022-23_1.pdf', '倪泽彬_Colby College_2022-23_1.pdf',
                 '倪泽彬_Colgate University_2022-23_1.pdf', '倪泽彬_Dickinson College_2022-23_1.pdf',
                 '倪泽彬_Franklin & Marshall College_2022-23_1.pdf', '倪泽彬_Kenyon College_2022-23_1.pdf',
                 '倪泽彬_Skidmore College_2022-23_1.pdf', '倪泽彬_Wesleyan University_2022-23_1.pdf',
                 '倪泽彬_Whitman College_2022-23_1.pdf', '傅小甜_Carnegie Mellon University_2022-23_2.pdf',
                 '傅小甜_Cornell University_2022-23_2.pdf', '傅小甜_University of Hong Kong_2022-23_2.pdf',
                 '傅小甜_University of Richmond_2022-23_2.pdf', '尹浩扬_Emory University_2021-22_1.pdf',
                 '尹荷清_University of Pennsylvania_2022-23_transfer_1.pdf']
    failed_file_list = []
    df_list = []
    for idx, file_name in enumerate(file_list):
        try:
            chi_name = file_name.split('_')[0]
            university = file_name.split('_')[1]
            if university.lower() == 'commonapp':
                continue
            file_path = os.path.join(pdf_directory, file_name)
            result = handle_single_pdf(file_path)
            result.update({'chi_name': chi_name, 'file_name': file_name, 'university': university})
            result = pd.Series(result).to_frame().T
            df_list.append(result)
            logger.info(f'file list index: {idx}, parsed {chi_name} doc: {file_name}')
            logger.info(f"\n{result}")
            sleep(10)
        except Exception as e:
            logger.exception(f'parse {chi_name} document "{file_name}" failed: {str(e)}')
            failed_file_list.append(file_name)
            continue

    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df[['chi_name', 'file_name', 'college', 'major', 'second_major']]
    result_file = f'major_result_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    final_df.to_csv(result_file)
    logger.info(f'total {len(failed_file_list)} files failed')
    logger.info(f'failed file list are: \n{failed_file_list}')


def handle_response(response):
    res = re.findall(r'"(.*)"', response.response)
    return dict(zip(['college', 'major', 'second_major'], res))


def handle_single_pdf(file_path):
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    format_instructions = 'The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "\\`\\`\\`json" and "\\`\\`\\`":\n\n```json\n[\n\tstring  // the college of the university which the student is applying to\n\tstring  // the major which the student is applying to\n\tstring  // the second major which the student is applying to\n]\n```'
    prompt = f"Use the following context to answer the question at the end. If you don't know the answer, just return empty string. \n{format_instructions}\n\n" + "        {context_str}\n\n        Question: {query_str}\n        Answer:"

    QA_PROMPT = QuestionAnswerPrompt(prompt)
    query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)
    response = query_engine.query("Which university, college and major is the student applying to?")
    return handle_response(response)


if __name__ == '__main__':
    # main()
    first_result = pd.read_csv('/Users/chengyu/stoooges/gpt_test/major_result_2023-05-09_20-29-21.csv', index_col=0)
    print(first_result)
    second_result = pd.read_csv('/Users/chengyu/stoooges/gpt_test/major_result_2023-05-11_15-49-32.csv', index_col=0)
    for _, row in second_result.iterrows():
        print(row)
        chi_name = row.loc['chi_name']
        college = row.loc['college']
        if first_result[(first_result['chi_name'] == chi_name) & (first_result['college'] == college)].empty:
            first_result = pd.concat([first_result, row.to_frame().T], ignore_index=True)

    first_result.sort_values(['chi_name', 'college'], ignore_index=True, inplace=True)
    result_file = f'major_result_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    first_result.to_csv(result_file)
    print(first_result)

