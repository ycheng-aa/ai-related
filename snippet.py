import os

from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate


os.environ["OPENAI_API_KEY"] = "sk-qlpEGGI7LgjDExHH03hZT3BlbkFJ1RT7eqBTjgF3GX7J0Xl"

def test():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "The following is a friendly conversation between a human and an AI. "
            "The AI is talkative and provides lots of specific details from its context."
            " If the AI does not know the answer to a question, it truthfully says it does not know."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    result = conversation.predict(input="Hi there!")
    print(result)
    # -> 'Hello! How can I assist you today?'

    result = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
    print(result)
    # -> "That sounds like fun! I'm happy to chat with you. Is there anything specific you'd like to talk about?"

    result = conversation.predict(input="Tell me about yourself.")
    print(result)


if __name__ == '__main__':
    # test()
    a = ['万熙培_University of Pennsylvania_2022-23_1.pdf', '万熙培_Washington University in St.Louis_2022-23_1.pdf', '严一苇_Cornell University_2021-22_1.pdf', '严一苇_Cornell University_2021-22_2.pdf', '严嘉仪_Cornell University_2021-22_2.pdf', '严嘉仪_Duke University_2021-22_2.pdf', '严嘉仪_Johns Hopkins University_2021-22_2.pdf', '严嘉仪_University of Southern California_2021-22_2.pdf', '严嘉仪_Washington University in St.Louis_2021-22_2.pdf', '于婧蕾_Boston College_2021-22_2.pdf', '于宗琪_Brown University_2021-22_transfer_1.pdf', '于宗琪_Columbia University_2022-23_transfer_1.pdf', '于宗琪_Cornell University_2021-22_transfer_1.pdf', '于宗琪_Duke University_2021-22_transfer_1.pdf', '于宗琪_Harvard University_2021-22_transfer_1.pdf', '于宗琪_Northwestern University_2021-22_transfer_1.pdf', '于宗琪_Stanford University_2021-22_transfer_1.pdf', '于宗琪_University of Chicago_2022-23_transfer_1.pdf', '于宗琪_University of Pennsylvania_2021-22_transfer_1.pdf', '于宗琪_Yale University_2021-22_transfer_1.pdf', '于馨淼_Barnard College_2022-23_2.pdf', '于馨淼_Carnegie Mellon University_2022-23_2.pdf', '于馨淼_Cornell University_2022-23_1.pdf', '于馨淼_Duke University_2022-23_1.pdf', '于馨淼_Rice University_2022-23_2.pdf', '于馨淼_University of Southern California_2022-23_1.pdf', '付晨溪_Rice University_2022-23_2.pdf', '付晨溪_University of Southern California_2022-23_2.pdf', '付晨溪_Washington University in St.Louis_2022-23_2.pdf', '付毅成_Cornell University_2022-23_transfer_1.pdf', '付毅成_University of Michigan-Ann Arbor_2022-23_transfer_1.pdf', '付毅成_Washington University in St.Louis_2022-23_transfer_1.pdf', '任越麒_Carnegie Mellon University_2022-23_2.pdf', '任越麒_University of Rochester_2022-23_2.pdf', '任越麒_University of Southern California_2022-23_2.pdf', '任越麒_Washington University in St.Louis_2022-23_2.pdf', '何嘉乐_Northeastern University_2022-23_1.pdf', '何嘉乐_Pennsylvania State University_2022-23_1.pdf', '何山_New York University_2022-23_1.pdf', '何山_Ohio State University_2022-23_1.pdf', '何山_UC_2022-23.pdf', '余懿炜_Brown University_2021-22_2.pdf', '余懿炜_Carnegie Mellon University_2021-22_1.pdf', '余懿炜_Cornell University_2021-22_1.pdf', '余懿炜_Duke University_2021-22_2.pdf', '余懿炜_University of Pennsylvania_2021-22_2.pdf', '余懿炜_Washington University in St.Louis_2021-22_1.pdf', '余懿炜_Wellesley College_2021-22_2.pdf', '侯妙严_Amherst College_2022-23_2.pdf', '侯妙严_Duke University_2022-23_1.pdf', '侯妙严_Harvard University_2022-23_1.pdf', '侯妙严_Smith College_2022-23_1.pdf', '侯笑恬_Barnard College_2022-23_2.pdf', '侯笑恬_Brown University_2022-23_2.pdf', '侯笑恬_Cornell University_2022-23_2.pdf', '俞泓辰_McGill University_2021-22_1.pdf', '倪泽彬_Bates College_2022-23_1.pdf', '倪泽彬_Colby College_2022-23_1.pdf', '倪泽彬_Colgate University_2022-23_1.pdf', '倪泽彬_Dickinson College_2022-23_1.pdf', '倪泽彬_Franklin & Marshall College_2022-23_1.pdf', '倪泽彬_Kenyon College_2022-23_1.pdf', '倪泽彬_Skidmore College_2022-23_1.pdf', '倪泽彬_Wesleyan University_2022-23_1.pdf', '倪泽彬_Whitman College_2022-23_1.pdf', '傅小甜_Carnegie Mellon University_2022-23_2.pdf', '傅小甜_Cornell University_2022-23_2.pdf', '傅小甜_University of Hong Kong_2022-23_2.pdf', '傅小甜_University of Richmond_2022-23_2.pdf', '尹浩扬_Emory University_2021-22_1.pdf', '尹荷清_University of Pennsylvania_2022-23_transfer_1.pdf']
    print(len(a))
