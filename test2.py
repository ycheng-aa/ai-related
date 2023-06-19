import os

from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import messages_to_dict, messages_from_dict
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from langchain.chains import LLMChain
from langchain.chains.base import Chain

from typing import Dict, List

# 小东
os.environ["OPENAI_API_KEY"] = "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"

# 公司
# os.environ["OPENAI_API_KEY"] = "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"
from langchain.vectorstores import Chroma

os.environ["SERPAPI_API_KEY"] = "de3e259fcac004c1116d823b3cb517fb24df19dc30e0d9486fd9b0644e6582c7"

def test_1():
    loader = TextLoader('state_of_the_union.txt', encoding='utf8')
    index = VectorstoreIndexCreator().from_loaders([loader])

    query = "What did the president say about Ketanji Brown Jackson"
    # print(index.query(query))

    print(index.query_with_sources(query))
    print(index.vectorstore.as_retriever())
    """
    VectorstoreIndexCreator is just a wrapper around all this logic. It is configurable in the text splitter it uses, the embeddings it uses, and the vectorstore it uses. For example, you can configure it as below:

    index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma, 
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
)
    """


def test_2():
    loader = TextLoader('state_of_the_union.txt', encoding='utf8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    db = Chroma.from_documents(texts, OpenAIEmbeddings())
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    query = "What did the president say about Ketanji Brown Jackson"
    print(qa.run(query))


def test_3():
    history = ChatMessageHistory()

    history.add_user_message("hi!")

    history.add_ai_message("whats up?")

    memory = ConversationBufferMemory()
    memory.chat_memory.add_user_message("hi!")
    memory.chat_memory.add_ai_message("whats up?")
    print(memory.load_memory_variables({}))

    print("----------------------------------------------------")

    memory = ConversationBufferMemory(return_messages=True)
    memory.chat_memory.add_user_message("hi!")
    memory.chat_memory.add_ai_message("whats up?")

    print(memory.load_memory_variables({}))


def test_4():
    llm = OpenAI(temperature=0)
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory()
    )
    print(conversation.predict(input="Hi there!"))
    print(conversation.predict(input="I'm doing well! Just having a conversation with an AI."))
    print(conversation.predict(input="Tell me about yourself."))


def test_5():
    history = ChatMessageHistory()

    history.add_user_message("hi!")

    history.add_ai_message("whats up?")

    print(history.messages)

    dicts = messages_to_dict(history.messages)

    print(dicts)

    new_messages = messages_from_dict(dicts)

    print(new_messages)


def test_6():
    from langchain.document_loaders import PyPDFLoader
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings

    loader = PyPDFLoader("/Users/chengyu/Downloads/tmp_preview_293/万熙培_Northeastern University_2022-23_1.pdf")
    index = VectorstoreIndexCreator().from_loaders([loader])
    query = "What is the student applying for?"
    print(index.query(query))


def test_7():
    chat = ChatOpenAI(temperature=0)
    conversation = ConversationChain(
        llm=chat,
        memory=ConversationBufferMemory()
    )

    print(conversation.run("Answer briefly. What are the first 3 colors of a rainbow?"))
    # -> The first three colors of a rainbow are red, orange, and yellow.
    print(conversation.run("And the next 4?"))
    # -> The next four colors of a rainbow are green, blue, indigo, and violet.


def test_8():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    second_prompt = PromptTemplate(
        input_variables=["company_name"],
        template="Write a catchphrase for the following company: {company_name}",
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt)

    overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

    catchphrase = overall_chain.run("colorful socks")

    print(catchphrase)


def test_9():
    response_schemas = [
        ResponseSchema(name="answer", description="answer to the user's question"),
        ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    print(format_instructions)
    print("---------------------------------------------------------")

    chat_model = ChatOpenAI(temperature=0)
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "answer the users question as best as possible.\n{format_instructions}\n{question}")
        ],
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )
    _input = prompt.format_prompt(question="what's the capital of france")
    print(_input)
    print("---------------------------------------------------------")
    output = chat_model(_input.to_messages())
    print(output)
    print("---------------------------------------------------------")

    result = output_parser.parse(output.content)
    print(result)


def test_10():
    class ConcatenateChain(Chain):
        chain_1: LLMChain
        chain_2: LLMChain

        @property
        def input_keys(self) -> List[str]:
            # Union of the input keys of the two chains.
            all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
            return list(all_input_vars)

        @property
        def output_keys(self) -> List[str]:
            return ['concat_output']

        def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
            output_1 = self.chain_1.run(inputs)
            output_2 = self.chain_2.run(inputs)
            return {'concat_output': output_1 + output_2}

    llm = OpenAI(temperature=0.9)

    prompt_1 = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain_1 = LLMChain(llm=llm, prompt=prompt_1)

    prompt_2 = PromptTemplate(
        input_variables=["product"],
        template="What is a good slogan for a company that makes {product}?",
    )

    chain_2 = LLMChain(llm=llm, prompt=prompt_2)

    concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)
    concat_output = concat_chain.run("colorful socks")
    print(f"Concatenated output:\n{concat_output}")

if __name__ == '__main__':
    test_10()
