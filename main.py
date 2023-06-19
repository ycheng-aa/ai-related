from langchain import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
import os

from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

from langchain.schema import (
    HumanMessage,
    SystemMessage
)


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate, MessagesPlaceholder,
)

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.prompts import FewShotPromptTemplate, PromptTemplate

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate


# 小东
os.environ["OPENAI_API_KEY"] = "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"

# 公司
# os.environ["OPENAI_API_KEY"] = "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"


os.environ["SERPAPI_API_KEY"] = "de3e259fcac004c1116d823b3cb517fb24df19dc30e0d9486fd9b0644e6582c7"

llm = OpenAI(temperature=0.9, model_name='gpt-3.5-turbo')


def test1():
    # text = "What would be a good company name for a company that makes cell phones?"
    # print(llm(text))

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    print(prompt.format(product="colorful socks"))


def test2():
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run('good fishes'))


def test3():
    llm = OpenAI(temperature=0)
    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Now let's test it out!
    agent.run(
        "What was the high temperature in SF yesterday in Fahrenheit? What is that number plus 100?")


def test4():
    llm = OpenAI(temperature=0)
    conversation = ConversationChain(llm=llm, verbose=True)
    output = conversation.predict(input="Hi there!")
    print(output)
    output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
    print(output)


def test5():
    chat = ChatOpenAI(temperature=0)
    # messages = [
    #     SystemMessage(content="You are a helpful assistant that translates English to Chinese."),
    #     HumanMessage(content="Translate this sentence from English to Chinese. I love programming.")
    #     ]
    # result = chat(messages)

    batch_messages = [
        [
            SystemMessage(content="You are a helpful assistant that translates English to Chinese."),
            HumanMessage(content="Translate this sentence from English to Chinese. I love programming.")
        ],
        [
            SystemMessage(content="You are a helpful assistant that translates English to Chinese."),
            HumanMessage(content="Translate this sentence from English to Chinese. I love artificial intelligence.")
        ],
    ]
    result = chat.generate(batch_messages)
    print(result)
    print(result.llm_output['token_usage'])


def test6():
    chat = ChatOpenAI(temperature=0)

    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # get a chat completion from the formatted messages
    print(chat(chat_prompt.format_prompt(input_language="English", output_language="Chinese",
                                         text="I love programming.").to_messages()))


def test7():
    chat = ChatOpenAI(temperature=0)

    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(input_language="English", output_language="Chinese", text="I love programming.")
    print(result)


def test8():
    # First, let's load the language model we're going to use to control the agent.
    chat = ChatOpenAI(temperature=0)

    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Now let's test it out!
    agent.run("Who is American president? What is his current age plus 100?")


def test9():
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


def test10():
    # First, create the list of few shot examples.
    examples = [
        {"word": "happy", "antonym": "sad"},
        {"word": "tall", "antonym": "short"},
    ]

    # Next, we specify the template to format the examples we have provided.
    # We use the `PromptTemplate` class for this.
    example_formatter_template = """
    Word: {word}
    Antonym: {antonym}\n
    """
    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template=example_formatter_template,
    )

    # Finally, we create the `FewShotPromptTemplate` object.
    few_shot_prompt = FewShotPromptTemplate(
        # These are the examples we want to insert into the prompt.
        examples=examples,
        # This is how we want to format the examples when we insert them into the prompt.
        example_prompt=example_prompt,
        # The prefix is some text that goes before the examples in the prompt.
        # Usually, this consists of intructions.
        prefix="Give the antonym of every input",
        # The suffix is some text that goes after the examples in the prompt.
        # Usually, this is where the user input will go
        suffix="Word: {input}\nAntonym:",
        # The input variables are the variables that the overall prompt expects.
        input_variables=["input"],
        # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
        example_separator="\n\n",
    )

    # We can now generate a prompt using the `format` method.
    print(few_shot_prompt.format(input="big"))


def test_11():
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )

    # These are a lot of examples of a pretend task of creating antonyms.
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"},
    ]

    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
        # This is the list of examples available to select from.
        examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        OpenAIEmbeddings(),
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        FAISS,
        # This is the number of examples to produce.
        k=2
    )
    mmr_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input",
        suffix="Input: {adjective}\nOutput:",
        input_variables=["adjective"],
    )
    # Input is a feeling, so should select the happy/sad example as the first one
    print(mmr_prompt.format(adjective="worried"))


def test_12():
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )

    # These are a lot of examples of a pretend task of creating antonyms.
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"},
    ]

    # These are examples of a fictional translation task.
    examples = [
        {"input": "See Spot run.", "output": "Ver correr a Spot."},
        {"input": "My dog barks.", "output": "Mi perro ladra."},
        {"input": "Spot can run.", "output": "Spot puede correr."},
    ]

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )
    example_selector = NGramOverlapExampleSelector(
        # These are the examples it has available to choose from.
        examples=examples,
        # This is the PromptTemplate being used to format the examples.
        example_prompt=example_prompt,
        # This is the threshold, at which selector stops.
        # It is set to -1.0 by default.
        threshold=-1.0,
        # For negative threshold:
        # Selector sorts examples by ngram overlap score, and excludes none.
        # For threshold greater than 1.0:
        # Selector excludes all examples, and returns an empty list.
        # For threshold equal to 0.0:
        # Selector sorts examples by ngram overlap score,
        # and excludes those with no ngram overlap with input.
    )
    dynamic_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the Spanish translation of every input",
        suffix="Input: {sentence}\nOutput:",
        input_variables=["sentence"],
    )

    # An example input with large ngram overlap with "Spot can run."
    # and no overlap with "My dog barks."
    print(dynamic_prompt.format(sentence="Spot can run fast."))
    print("---------------------------------------------------------------")

    # You can add examples to NGramOverlapExampleSelector as well.
    new_example = {"input": "Spot plays fetch.", "output": "Spot juega a buscar."}

    example_selector.add_example(new_example)
    print(dynamic_prompt.format(sentence="Spot can run fast."))

    print("---------------------------------------------------------------")
    # You can set a threshold at which examples are excluded.
    # For example, setting threshold equal to 0.0
    # excludes examples with no ngram overlaps with input.
    # Since "My dog barks." has no ngram overlaps with "Spot can run fast."
    # it is excluded.
    example_selector.threshold = 0.0
    print(dynamic_prompt.format(sentence="Spot can run fast."))

    print("---------------------------------------------------------------")

    # Setting small nonzero threshold
    example_selector.threshold = 0.09
    print(dynamic_prompt.format(sentence="Spot can play fetch."))

    print("---------------------------------------------------------------")

    # Setting threshold greater than 1.0
    example_selector.threshold = 1.0 + 1e-9
    print(dynamic_prompt.format(sentence="Spot can play fetch."))


def test_13():
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )

    # These are a lot of examples of a pretend task of creating antonyms.
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"},
    ]
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # This is the list of examples available to select from.
        examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        OpenAIEmbeddings(),
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        Chroma,
        # This is the number of examples to produce.
        k=1
    )
    similar_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input",
        suffix="Input: {adjective}\nOutput:",
        input_variables=["adjective"],
    )

    # Input is a feeling, so should select the happy/sad example
    print(similar_prompt.format(adjective="worried"))

    print("---------------------------------------------------------------")

    # Input is a measurement, so should select the tall/short example
    print(similar_prompt.format(adjective="fat"))

    print("---------------------------------------------------------------")

    similar_prompt.example_selector.add_example({"input": "enthusiastic", "output": "apathetic"})
    print(similar_prompt.format(adjective="joyful"))



if __name__ == '__main__':
    test_13()
