# (cf) https://python.langchain.com/docs/integrations/document_loaders/microsoft_word/
# (cf) https://python.langchain.com/docs/integrations/document_loaders/microsoft_excel/

import os

import chainlit as cl
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import (Docx2txtLoader, PyMuPDFLoader,
                                        UnstructuredExcelLoader)
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage


def pdf_loader(file_path):
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    return data


def word_loader(file_path):
    loader = Docx2txtLoader(file_path)
    data = loader.load()
    return data


def excel_loader(file_path):
    loader = UnstructuredExcelLoader(file_path, mode="elements")
    data = loader.load()
    return data

    
ALLOWED_MIME_TYPES = [
    "application/pdf",
    "application/octet-stream",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
]

ALLOWED_EXTENSIONS = [
    ".pdf",
    ".docx",
    ".xlsx",
]

MAX_NUM_CHAR = 5000
MAX_SIZE_MB = 20


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file (await)
    while files == None:
        files = await cl.AskFileMessage(
            max_size_mb=MAX_SIZE_MB,
            content="Please upload a PDF/WORD/EXCEL file to begin!",
            accept=ALLOWED_MIME_TYPES,
            raise_on_timeout=False,
        ).send()

    file = files[0]
    ext = os.path.splitext(file.name)[1]

    # Load the file
    if ext in ALLOWED_EXTENSIONS:
        if ext == ".pdf":
            documents = pdf_loader(file.path)
        elif ext == ".docx":
            documents = word_loader(file.path)
        elif ext == ".xlsx":
            documents = excel_loader(file.path)

        # Merge contents that are divided into pages.
        content = ""
        for document in documents:
            content += document.page_content
        cl.user_session.set("document", content[:MAX_NUM_CHAR])    # Due to the limitation of the number of tokens

        await cl.Message(
            content=f"`{file.name}` uploaded, it contains {len(content)} characters!"
        ).send()

    else:
        await cl.Message(
            content="The format of uploaded file is not supported. Please upload word/excel/pdf files."
        ).send()


@cl.on_message
async def on_message(question: cl.Message):

    model = ChatOpenAI(model="gpt-4-0125-preview")
    document = cl.user_session.get("document")
    prompt = PromptTemplate(
        template="Please answer the questions based on the document.\ndocument:\n{document}\nquestion:\n{question}",
        input_variables=["document", "question"],
    )

    # Get answers (not by streaming)
    answer = model(
        [
            HumanMessage(
                content=prompt.format(
                    document=document,
                    question=question.content,
                )
            )
        ]
    ).content

    # Print answers
    await cl.Message(content=answer).send()

