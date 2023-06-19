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

# 小东
os.environ["OPENAI_API_KEY"] = "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"

# 公司
os.environ["OPENAI_API_KEY"] = "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"

os.environ["SERPAPI_API_KEY"] = "de3e259fcac004c1116d823b3cb517fb24df19dc30e0d9486fd9b0644e6582c7"


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(in_data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(in_data, f)


def generate_prompt_template():
    response_schemas = [
        ResponseSchema(name="abbre", description="the English abbreviation of the input competition name"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = 'The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "\\`\\`\\`json" and "\\`\\`\\`":\n\n```json\n[\n\tstring  // the English abbreviation of the input competition name\n]\n```'
    template = f"Answer the users question as best as possible at the end. If you don't know the answer, just return empty string, don't try to make up an answer. {format_instructions}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "what is the official English abbreviation for the student competition named {competition_name}?"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt, output_parser


def get_abbreviation(in_name):
    chat_prompt, output_parser = generate_prompt_template()
    # in_name = 'Green Brain of the Year" International Concept Project Competition-High School Students'
    chat = ChatOpenAI(temperature=0.7)
    # prompt = PromptTemplate(
    #     input_variables=["competition_name"],
    #     template="what is the English abbreviation for the student competition named {competition_name}?",
    # )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(competition_name=in_name)
    result = re.findall(r'\[\s*"(.+)"\s*\]', response)
    if result:
        result = result[0]
    else:
        result = None
        logger.error(f'{in_name}没有查出结果，gpt原始返回为：{response}')

    return result


def agent(in_name):
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Now let's test it out!
    result = agent.run(
        f"{in_name} 的英文简称是什么?")
    print("google search 做为 agent结果")
    print(result)
    print("-----------------\n\n")
    return result


def main():
    data = load_pickle('/Users/chengyu/stoooges/gpt_test/data/competition_data')
    competition_and_id_list = []

    for i in data:
        competition_name = i.get('fields', {}).get('比赛全称')
        competition_id = i.get('fields', {}).get('编号')
        competition_and_id_list.append([competition_name, competition_id])

    ori_name_list = []
    abbreviation_list = []
    failed_list = []
    competition_id_list = []
    for idx, competition_name_id in enumerate(competition_and_id_list):
        try:
            competition_name = competition_name_id[0]
            competition_id = competition_name_id[1]
            abbreviation = get_abbreviation(competition_name)
            # abbreviation = agent(competition_name)
            if not abbreviation:
                raise ValueError(f'没有获得 {competition_name} 的简写结果')
            abbreviation_list.append(abbreviation)
            ori_name_list.append(competition_name)
            competition_id_list.append(competition_id)
            logger.info(f'index {idx}, {competition_name}: {abbreviation}')
        except Exception as e:
            logger.exception(f'handle {competition_name} failed, {str(e)}')
            failed_list.append(competition_name)
        if idx > 0 and idx % 50 == 0:
            save_pickle([ori_name_list, abbreviation_list], 'name_and_abbreviation_result')
        sleep(25)

    df = pd.DataFrame.from_dict({'编号': competition_id_list, '竞赛名称': ori_name_list, '简称': abbreviation_list})
    print(df)
    result_file = f'competition_abbr_result_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    df.to_csv(result_file)




if __name__ == '__main__':
    main()
    # template = "You are a helpful assistant that translates {input_language} to {output_language}."
    # system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    # human_template = "{text}"
    # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    # pass
    # agent()
    # get_abbreviation('dfdf')

    # from langchain.llms import OpenAI, OpenAIChat
    #
    # llm = OpenAIChat()
    # question = """"面向学生的竞赛活动 "Green Brain of the Year" International Concept Project Competition-High School Students，它的常见简称是什么"""
    # print('OpenAIChat -----------------------')
    # print(llm(question))
