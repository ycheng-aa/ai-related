import os
import pickle
import re
from datetime import datetime
from time import sleep

import pandas as pd
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from logger import logger

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# 小东
# os.environ["OPENAI_API_KEY"] = "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"

# 公司
os.environ["OPENAI_API_KEY"] = "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"

os.environ["SERPAPI_API_KEY"] = "de3e259fcac004c1116d823b3cb517fb24df19dc30e0d9486fd9b0644e6582c7"


def generate_prompt_template():
    response_schemas = [
        ResponseSchema(name="abbre", description="the English abbreviation of the input summer school program"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = 'The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "\\`\\`\\`json" and "\\`\\`\\`":\n\n```json\n[\n\tstring  // the English abbreviation of the input competition name\n]\n```'
    template = f"Answer the users question as best as possible at the end. If you don't know the answer, just return empty string, don't try to make up an answer. {format_instructions}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "what is the English abbreviation for the summer school program named {summerschool_name} in the institution {university_name}?"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt, output_parser


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(in_data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(in_data, f)


def get_abbreviation(in_name, university_name, chat):
    chat_prompt, output_parser = generate_prompt_template()
    # prompt = PromptTemplate(
    #     input_variables=["summerschool_name"],
    #     template="what is the English abbreviation for the summer school program named {summerschool_name}?",
    # )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(summerschool_name=in_name, university_name=university_name)
    result = re.findall(r'\[\s*"(.+)"\s*\]', response)
    if result:
        result = result[0]
    else:
        result = None
        logger.error(f'学校{university_name}的{in_name}没有查出结果，gpt原始返回为：{response}')

    return result

def main():
    data = load_pickle('/Users/chengyu/stoooges/gpt_test/data/summerschool_data')
    summerschool_and_id_list = []

    for i in data:
        summerschool_name = i.get('fields', {}).get('夏校名称')
        summerschool_id = i.get('fields', {}).get('编号')
        university_name = i.get('fields', {}).get('学校')
        summerschool_and_id_list.append([summerschool_name, summerschool_id, university_name])

    max_idx = len(summerschool_and_id_list)
    chat = ChatOpenAI(temperature=0.7)
    ori_name_list = []
    abbreviation_list = []
    failed_list = []
    summerschool_id_list = []
    university_name_list = []
    for idx, competition_name_id in enumerate(summerschool_and_id_list):
        try:
            summerschool = competition_name_id[0]
            summerschool_id = competition_name_id[1]
            university_name = competition_name_id[2]
            abbreviation = get_abbreviation(summerschool, university_name, chat)
            # abbreviation = agent(summerschool)
            if not abbreviation:
                raise ValueError(f'没有获得 {summerschool} 的简写结果')
            abbreviation_list.append(abbreviation)
            ori_name_list.append(summerschool)
            university_name_list.append(university_name)
            summerschool_id_list.append(summerschool_id)
            logger.info(f'index {idx}/{max_idx}, {university_name}的{summerschool}: {abbreviation}')
        except Exception as e:
            logger.exception(f'handle {university_name} 的 {summerschool} failed, {str(e)}')
            failed_list.append(summerschool)
        if idx > 0 and idx % 50 == 0:
            save_pickle([ori_name_list, abbreviation_list], 'summerschool_name_and_abbreviation_result')
        sleep(8)

    df = pd.DataFrame.from_dict({'编号': summerschool_id_list, '夏校名称': ori_name_list, '简称': abbreviation_list,
                                 '学校': university_name_list})
    print(df)
    result_file = f'./data/summerschool_abbr_result_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    df.to_csv(result_file)


if __name__ == '__main__':
    main()
