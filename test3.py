import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory

from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

os.environ["OPENAI_API_KEY"] = "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"

# 公司
# os.environ["OPENAI_API_KEY"] = "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"



def test_1():
    """
    ConversationalRetrievalChain使用

    :return:
    """
    loader = TextLoader("state_of_the_union.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)

    query = "What did the president say about Ketanji Brown Jackson"
    result = qa({"question": query})
    print(result["answer"])
    print("----------------------------------------------------")

    print(memory)
    print("----------------------------------------------------")

    query = "Did he mention who she suceeded"
    result = qa({"question": query})

    print(result)
    print("----------------------------------------------------")
    print(result["answer"])


def test_2():
    """
    pass memory in explicitly

    :return:
    """
    loader = TextLoader("state_of_the_union.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever())

    chat_history = []
    query = "What did the president say about Ketanji Brown Jackson"
    result = qa({"question": query, "chat_history": chat_history})

    print(result["answer"])
    print("----------------------------------------------------")

    chat_history = [(query, result["answer"])]
    query = "Did he mention who she suceeded"
    print(chat_history)
    print("----------------------------------------------------")
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])


def test_3():
    """
    Return Source Documents

    :return:
    """
    loader = TextLoader("state_of_the_union.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(),
                                               return_source_documents=True)
    chat_history = []
    query = "What did the president say about Ketanji Brown Jackson"
    result = qa({"question": query, "chat_history": chat_history})

    print(result)
    print("----------------------------------------------------")
    print(result['source_documents'])
    print("----------------------------------------------------")
    print(result['source_documents'][0])


def test_4():
    """
    ConversationalRetrievalChain with map_reduce

    :return:
    """
    loader = TextLoader("state_of_the_union.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    llm = OpenAI(temperature=0)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm, chain_type="map_reduce")

    chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )

    chat_history = []
    query = "What did the president say about Ketanji Brown Jackson"
    result = chain({"question": query, "chat_history": chat_history})

    print(result['answer'])


def test_5():
    loader = TextLoader("state_of_the_union.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    llm = OpenAI(temperature=0)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

    chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )

    chat_history = []
    query = "What did the president say about Ketanji Brown Jackson"
    result = chain({"question": query, "chat_history": chat_history})

    print(result['answer'])


def test_6():
    """
    ConversationalRetrievalChain with streaming to stdout, 运行不通

    :return:
    """
    loader = TextLoader("state_of_the_union.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    llm = OpenAI(temperature=0)
    streaming_llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=doc_chain, question_generator=question_generator)

    chat_history = []
    query = "What did the president say about Ketanji Brown Jackson"
    result = qa({"question": query, "chat_history": chat_history})
    print(question_generator)

    chat_history = [(query, result["answer"])]
    query = "Did he mention who she suceeded"
    result = qa({"question": query, "chat_history": chat_history})
    print(question_generator)


def test_7():
    def get_chat_history(inputs) -> str:
        res = []
        for human, ai in inputs:
            res.append(f"Human:{human}\nAI:{ai}")
        return "\n".join(res)

    loader = TextLoader("state_of_the_union.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(),
                                               get_chat_history=get_chat_history)

    chat_history = []
    query = "What did the president say about Ketanji Brown Jackson"
    result = qa({"question": query, "chat_history": chat_history})

    print(result['answer'])

    chat_history = [(query, result["answer"])]
    query = "Did he mention who she suceeded"
    print(chat_history)
    print("----------------------------------------------------")
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])

if __name__ == '__main__':
    test_7()


