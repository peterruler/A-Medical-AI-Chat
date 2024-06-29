from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    
    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path="mistral-logo.png")
    ]
    await cl.Message(content="Hello, ich bin Mistral. Wie kann ich dir helfen ?", elements=elements).send()
    model = Ollama(model="mistral")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Du bist ein hilfsbereiter Bot, der Texte liest und Fragen dazu beantwortet.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):

    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()