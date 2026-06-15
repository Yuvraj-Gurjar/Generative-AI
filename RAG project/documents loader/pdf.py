# # for token text splitting technique

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter

# data=PyPDFLoader("documents loader/RAG.pdf")

# docs=data.load()

# splitter=TokenTextSplitter(
#     chunk_size=100,
#     chunk_overlap=1
# )

# chunks=splitter.split_documents(docs)

# # print(docs[2])
# # print(len(chunks))
# # to see the first chunk
# print(chunks[0].page_content)


# for recursive text splitting technique
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

data=PyPDFLoader("documents loader/RAG.pdf")

docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=1
)

chunks=splitter.split_documents(docs)

# print(docs[2])
# print(len(chunks))
# to see the first chunk
print(chunks[0].page_content)