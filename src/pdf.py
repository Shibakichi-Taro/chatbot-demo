# (cf) LangChain PDF: https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf/#using-pymupdf
# (cf) PyMuPDF: https://pymupdf.readthedocs.io/en/latest/app1.html#plain-text/
# (cf) Chainlit Ask APIs: https://docs.chainlit.io/advanced-features/ask-user
#   (cf) Chainlit AskFileMessage: https://docs.chainlit.io/api-reference/ask/ask-for-file#example (else) 
"""
* PyMuPDF の選定理由

|                | PyMuPDF | PyPDF2 | PDFMiner |
|----------------|---------|--------|----------|
|テキスト抽出    |o        |o (*1)  |o         |
|ブックマーク抽出|o        |x       |o         |
|画像抽出        |o        |o       |o (*2)    |
|複数ファイル結合|o        |o       |x         |
|ページ分割      |o        |o       |x         |

(*1) 日本語非対応
(*2) JPEGのみ

"""

import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

MAX_NUM_CHAR = 5000
MAX_SIZE_MB = 20

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file (await)
    while files == None:
        files = await cl.AskFileMessage(
            max_size_mb=MAX_SIZE_MB,
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()

    file = files[0]
    documents = PyMuPDFLoader(file.path).load()

    # Merge contents that are divided into pages.
    content = ""
    for document in documents:
        content += document.page_content
    cl.user_session.set("document", content[:MAX_NUM_CHAR])    # Due to the limitation of the number of tokens

    await cl.Message(
        content=f"`{file.name}` uploaded, it contains {len(content)} characters!"
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