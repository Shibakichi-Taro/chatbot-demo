# (cf) https://docs.chainlit.io/integrations/langchain#with-langchain-expression-language-lcel
# (cf) decorator: https://docs.chainlit.io/concepts/chat-lifecycle
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    """ Sets up an instance of `Runnable` for each chat session.
        The `Runnable` is invoked everytime a user sends a message to generate the response.
        The `@on_chat_start` decorator is used to define a hook that is called when a new chat session is created.
    """
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)   # User Session (cf) https://docs.chainlit.io/concepts/user-session


@cl.on_message
async def on_message(question: cl.Message):
    """ The callback handler is responsible for listening to the chain’s intermediate steps and sending them to the UI.
        The `@on_message` decorator is used to define a hook that is called when a new message is received from the user.
    """
    runnable = cl.user_session.get("runnable")  # User Session (cf) https://docs.chainlit.io/concepts/user-session

    # Get answers (by streaming) (cf) https://docs.chainlit.io/advanced-features/streaming
    answer = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": question.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await answer.stream_token(chunk)

    # Print answers
    await answer.send()