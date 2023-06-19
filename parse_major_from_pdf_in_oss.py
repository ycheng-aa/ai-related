import os
from time import sleep

import pandas as pd
import re

from langchain import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from logger import logger
from utils import get_basic_auth_file
from utils import write_content_to_file, OSS, langchain_handle_single_document, llama_handle_single_pdf, save_pickle


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# 公司
os.environ["OPENAI_API_KEY"] = "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"

# os.environ["OPENAI_API_KEY"] = "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"

TMP_IDR = os.environ.get('OSS_TMP_LOCAL_DIR', '/Users/chengyu/stoooges/gpt_test/data')


def _parse_oss_file_links(file_link_str):
    result = re.findall(r'https://stoooges-test.oss-cn-beijing.aliyuncs.com/([^\.]+\.\w+)', file_link_str)
    if result and (not result[0].endswith('.pdf') and not result[0].endswith('.docx')):
        result = re.findall(r'https://stoooges-test.oss-cn-beijing.aliyuncs.com/(.+\.(?:pdf|docx))', file_link_str)

    return result


def _parse_static_file_links(file_link_str):
    result = re.findall(r'https://test.stoooges.cn/static/universityFiles/([^\.]+\.\w+)', file_link_str)
    if result and (not result[0].endswith('.pdf') and not result[0].endswith('.docx')):
        result = re.findall(r'https://test.stoooges.cn/static/universityFiles/(.+\.(?:pdf|docx))', file_link_str)

    return result


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


def generate_transfer_prompt():
    prompt_template = """Use the following context to answer the question at the end. If you don't know the answer, just return empty string. {format_instructions}

        {context}

        Question: {question}
        Answer:"""  # noqa

    response_schemas = [
        ResponseSchema(name="college", description="the college of the university which the student is applying to"),
        ResponseSchema(name="major", description="the major which the student is applying to"),
        ResponseSchema(name="second_major", description="the second major which the student is applying to"),
        ResponseSchema(name="third_major", description="the third major which the student is applying to")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"],
        partial_variables={"format_instructions": format_instructions}
    )
    return prompt, output_parser


def handle_llama_response(response):
    res = re.findall(r'"(.*)"', response.response)
    return dict(zip(['college', 'major', 'second_major'], res))


def generate_llama_prompt():
    format_instructions = 'The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "\\`\\`\\`json" and "\\`\\`\\`":\n\n```json\n[\n\tstring  // the college of the university which the student is applying to\n\tstring  // the major which the student is applying to\n\tstring  // the second major which the student is applying to\n]\n```'
    prompt = f"Use the following context to answer the question at the end. If you don't know the answer, just return empty string. \n{format_instructions}\n\n" + "        {context_str}\n\n        Question: {query_str}\n        Answer:"
    return prompt


def handle_transfer_llama_response(response):
    if not response.response:
        return {}
    res = re.findall(r'"(.*)"', response.response)
    return dict(zip(['college', 'major', 'second_major', 'third_major'], res))


def generate_transfer_llama_prompt():
    format_instructions = 'The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "\\`\\`\\`json" and "\\`\\`\\`":\n\n```json\n[\n\tstring  // the college or school to which the student is applying\n\tstring  // the major or academic interest to which the student is applying\n\tstring  // the second major or academic interest to which the student is applying\n\tstring  // the third major or academic interest to which the student is applying\n]\n```'
    prompt = f"Use the following context to answer the question at the end. If you don't know the answer, just return empty string. \n{format_instructions}\n\n" + "        {context_str}\n\n        Question: {query_str}\n        Answer:"
    return prompt


def main(feishu_file_path):
    feishu_file_df = pd.read_csv(feishu_file_path)
    prompt, output_parser = generate_prompt()
    llama_prompt = generate_llama_prompt()
    result_list = []
    total_student = feishu_file_df.shape[0]
    sleeping_secs = 25

    for idx, row in feishu_file_df.iterrows():
        app_year = row.loc['申请年份']
        chi_name = row.loc['学生姓名']
        app_university = row.loc['申请学校']
        file_link_str = row.loc['文件链接']
        line_no = idx + 1
        parsed_result = {}
        query = 'What college and major is the student applying to?'
        file_link_list = []
        if pd.isna(file_link_str) or not file_link_str:
            logger.info(f'{chi_name} 申请 {app_university} 没有oss文件链接, file link: {file_link_str}')
        else:
            file_link_list = _parse_oss_file_links(file_link_str)

        # 遍历该人该校下的所有文件, 首先使用langchain进行解析
        for f in file_link_list:
            tmp_file_path = None
            try:
                tmp_file_path = os.path.join(TMP_IDR, os.path.basename(f))
                write_content_to_file(OSS.bucket.get_object(f).read(), tmp_file_path)
                parsed_result = langchain_handle_single_document(tmp_file_path, prompt, output_parser, query)
                if parsed_result:
                    break
            except Exception as e:
                logger.info(f'failed to parse with LangChain, lineno: {line_no}, name: {chi_name}, file name: {f}')
                logger.exception(str(e))
            finally:
                sleep(sleeping_secs)
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

        # 失败后使用 llama 继续尝试一次
        if not parsed_result:
            for f in file_link_list:
                tmp_file_path = None
                try:
                    tmp_file_path = os.path.join(TMP_IDR, os.path.basename(f))
                    write_content_to_file(OSS.bucket.get_object(f).read(), tmp_file_path)
                    response = llama_handle_single_pdf(tmp_file_path, llama_prompt, query)
                    parsed_result = handle_llama_response(response)
                    if parsed_result:
                        break
                except Exception as e:
                    logger.info(f'failed to parse with Llama Index, lineno: {line_no}, name: {chi_name}, '
                                f'file name: {f}')
                    logger.exception(str(e))
                finally:
                    sleep(sleeping_secs)
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
        if not parsed_result:
            parsed_result = {'college': '', 'major': '', 'second_major': ''}
        parsed_result.update({'原始行号': line_no, '学生姓名': chi_name, '申请年份': app_year,
                              '申请学校': app_university})
        parsed_df = pd.Series(parsed_result).to_frame().T
        result_list.append(parsed_df)
        if idx > 0 and idx // 50 == 0:
            save_pickle(result_list, '/Users/chengyu/stoooges/gpt_test/data/2022本科补全_专业_result_list')

        logger.info(f'{line_no}, name: {chi_name}, file name: {f}: {parsed_result}')
        logger.info(f'处理完{line_no} / {total_student}')

    # 处理UCAS
    for idx, row in feishu_file_df.iterrows():
        app_year = row.loc['申请年份']
        chi_name = row.loc['学生姓名']
        app_university = row.loc['申请学校']
        file_link_str = row.loc['文件链接']
        line_no = idx + 1
        query = f'What college and major is the student applying to in {app_university}?'
        if pd.isna(file_link_str) or not file_link_str or _parse_oss_file_links(file_link_str):
            continue
        file_link_list = _parse_static_file_links(file_link_str)
        if not file_link_list:
            continue
        f = file_link_list[0]
        tmp_file_path = os.path.join(TMP_IDR, os.path.basename(f))
        try:
            get_basic_auth_file(file_link_str, 'Stoooges', 'Stoooges2020!', tmp_file_path)
            parsed_result = langchain_handle_single_document(tmp_file_path, prompt, output_parser, query)
            if not parsed_result:
                parsed_result = {'college': '', 'major': '', 'second_major': ''}
            parsed_result.update({'原始行号': line_no, '学生姓名': chi_name, '申请年份': app_year,
                                  '申请学校': app_university})
            parsed_df = pd.Series(parsed_result).to_frame().T
            result_list.append(parsed_df)
            logger.info(f'{line_no}, name: {chi_name}, file name: {f}: {parsed_result}')
        except Exception as e:
            logger.exception(str(e))
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            sleep(sleeping_secs)

    result_df = pd.concat(result_list, ignore_index=True)
    result_df = result_df[['原始行号', '学生姓名', '申请年份', '申请学校', 'college', 'major', 'second_major']]
    result_df.to_csv('/Users/chengyu/stoooges/gpt_test/data/本科副表-学校和专业_2022本科补全_专业_UCAS.csv')


def main_transfer(feishu_file_path):
    feishu_file_df = pd.read_csv(feishu_file_path)
    llama_prompt = generate_transfer_llama_prompt()
    result_list = []
    total_student = feishu_file_df.shape[0]
    sleeping_secs = 8

    for idx, row in feishu_file_df.iterrows():
        app_year = row.loc['申请年份']
        chi_name = row.loc['学生姓名']
        app_university = row.loc['申请学校']
        file_link_str = row.loc['文件链接']
        line_no = idx + 1
        parsed_result = {}
        query = 'What college or school and majors or academic interests to which the student is applying to?'
        file_link_list = []
        is_oss = True
        if pd.isna(file_link_str) or not file_link_str:
            logger.info(f'{chi_name} 申请 {app_university} 没有oss文件链接, file link: {file_link_str}')
        else:
            file_link_list = _parse_oss_file_links(file_link_str)
            if not file_link_list:
                file_link_list = _parse_static_file_links(file_link_str)
                is_oss = False

            # # 遍历该人该校下的所有文件, 首先使用langchain进行解析
            # for f in file_link_list:
            #     tmp_file_path = None
            #     try:
            #         tmp_file_path = os.path.join(TMP_IDR, os.path.basename(f))
            #         if is_oss:
            #             write_content_to_file(OSS.bucket.get_object(f).read(), tmp_file_path)
            #         else:
            #             get_basic_auth_file(file_link_str, 'Stoooges', 'Stoooges2020!', tmp_file_path)
            #         parsed_result = langchain_handle_single_document(tmp_file_path, prompt, output_parser, query)
            #         if parsed_result:
            #             break
            #     except Exception as e:
            #         logger.info(
            #             f'failed to parse with LangChain, lineno: {line_no}, name: {chi_name}, file name: {f}')
            #         logger.exception(str(e))
            #     finally:
            #         if tmp_file_path and os.path.exists(tmp_file_path):
            #             os.remove(tmp_file_path)
            #         sleep(sleeping_secs)

        # 失败后使用 llama 继续尝试一次
        if not parsed_result:
            for f in file_link_list:
                tmp_file_path = None
                try:
                    tmp_file_path = os.path.join(TMP_IDR, os.path.basename(f))
                    if is_oss:
                        write_content_to_file(OSS.bucket.get_object(f).read(), tmp_file_path)
                    else:
                        get_basic_auth_file(file_link_str, 'Stoooges', 'Stoooges2020!', tmp_file_path)
                    response = llama_handle_single_pdf(tmp_file_path, llama_prompt, query)
                    parsed_result = handle_transfer_llama_response(response)
                    if parsed_result:
                        break
                except Exception as e:
                    logger.info(f'failed to parse with Llama Index, lineno: {line_no}, name: {chi_name}, '
                                f'file name: {f}')
                    logger.exception(str(e))
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
                    sleep(sleeping_secs)

        final_parsed_result = {'college': '', 'major': '', 'second_major': '', 'third_major': '', '原始行号': line_no,
                               '学生姓名': chi_name, '申请年份': app_year, '申请学校': app_university}
        final_parsed_result.update(parsed_result)

        parsed_df = pd.Series(final_parsed_result).to_frame().T
        result_list.append(parsed_df)
        if idx > 0 and idx // 50 == 0:
            save_pickle(result_list, '/Users/chengyu/stoooges/gpt_test/data/2022本科补全_专业_result_list')

        logger.info(f'{line_no}, name: {chi_name}, file name: {f}:\n{parsed_df}')
        logger.info(f'处理完{line_no} / {total_student}')
    save_pickle(result_list, '/Users/chengyu/stoooges/gpt_test/data/2022本科补全_专业_result_list')
    result_df = pd.concat(result_list, ignore_index=True)
    result_df = result_df[['原始行号', '学生姓名', '申请年份', '申请学校', 'college', 'major', 'second_major', 'third_major']]
    result_df.to_csv('/Users/chengyu/stoooges/gpt_test/data/转学副表-学校和专业_2023转学补全_专业.csv')





if __name__ == '__main__':
    # main('/Users/chengyu/stoooges/gpt_test/data/veglib卖菜库导出数据_本科副表-学校和专业_2022本科补全.csv')
    main_transfer('/Users/chengyu/stoooges/gpt_test/data/veglib卖菜库导出数据_转学副表-学校和专业_2023转学补全.csv')

    # print(_parse_oss_file_links('https://stoooges-test.oss-cn-beijing.aliyuncs.com/urmFile/赵宏林_Washington University in St.Louis_2023-24_685854.pdf'))
