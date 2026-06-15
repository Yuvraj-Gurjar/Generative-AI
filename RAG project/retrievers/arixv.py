import arxiv
from langchain_core.documents import Document

# 1. Create a client and query arxiv directly using their new API
client = arxiv.Client()
search = arxiv.Search(
    query="large language models",
    max_results=2,
)

docs = []
# 2. Fetch results and wrap them in LangChain Documents
for result in client.results(search):
    doc = Document(
        page_content=result.summary,
        metadata={
            "Title": result.title,
            # Extract author names and join them into a single string
            "Authors": ", ".join([author.name for author in result.authors]) 
        }
    )
    docs.append(doc)

# 3. Print results exactly as you had them before
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content[:500])