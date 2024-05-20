from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl
from googletrans import Translator


translator = Translator()

@cl.on_chat_start
async def on_chat_start():
    
    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path="gemma.jpeg")
    ]
    await cl.Message(content="Hello there, I am Gemma. How can I help you ?", elements=elements).send()
    model = Ollama(model="biomistral")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable doctor. Provide just the answer related to medical field.",
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
    message.content = translator.translate(message.content, src='de', dest='en').text
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
'''
@cl.on_message
async def main(message: str):
    # Übersetze die eingehende Nachricht vom Englischen ins Deutsche
    translated_response = translator.translate(message.content, src='en', dest='de').text

    # Sende die übersetzte Antwort als Antwort zurück
    await cl.Message(content=str(translated_response)).send()
'''