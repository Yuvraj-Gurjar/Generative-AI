# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.prompts import ChatPromptTemplate
# load_dotenv()
# # for notes.txt file
# # data=TextLoader("documents loader/notes.txt")

# # for RAG.pdf file
# data=PyPDFLoader("documents loader/RAG.pdf")
# docs=data.load()
# template=ChatPromptTemplate.from_messages(
#     [("system","You are the AI that summarizes text"),
#      ("human","{data}")]
# )
# model=ChatMistralAI(model="mistral-medium-3-5")
# prompt=template.format_messages(data=docs)

# result =model.invoke(prompt)
# print(result.content)

#-----------------------------------------------------------------------------------------------------------------

# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# load_dotenv()
# # for notes.txt file
# # data=TextLoader("documents loader/notes.txt")

# # for RAG.pdf file
# data=PyPDFLoader("documents loader/deeplearning.pdf")
# docs=data.load()

# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100
# )

# chunks=splitter.split_documents(docs)

# template=ChatPromptTemplate.from_messages(
#     [("system","You are the AI that summarizes text"),
#      ("human","{data}")]
# )
# model=ChatMistralAI(model="mistral-medium-3-5")

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model=MistralAIEmbeddings()

# to retrieve the data fro chroma db we use vector
vectorstore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
# now data loaded to the vector database now we have to retrieve so-
retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lamda_mult":0.5
    }
)

llm=ChatMistralAI(model="mistral-medium-3-5")

# now creating prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
) 

print("RAG system created")

print("Press 0 to exit")
while True:
    query=input("You:")
    if query=="0":
        break

    docs=retriever.invoke(query)
    context="\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })

    # now send this context and question to the llm
    response=llm.invoke(final_prompt)
    print(f"\n AI:{response.content}")