from langchain_community.document_loaders import TextLoader
# for character wise splitting 
from langchain_text_splitters import CharacterTextSplitter
data = TextLoader("documents loader/notes.txt")

splitter=CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

docs=data.load()

chunks=splitter.split_documents(docs)  #generate only one chunk if we print len(chunks) why? ans:-1
                                       #to get exact chunk u hav eto giv extra space in notes.txt ans:-2

print(len(chunks))
for i in chunks:
    print(i.page_content)
    print()
    print()
    print()