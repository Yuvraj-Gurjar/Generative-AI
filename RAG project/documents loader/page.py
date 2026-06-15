# load form web data
from langchain_community.document_loaders import WebBaseLoader

url="http://apple.com/shop/buy-mac/macbook-pro/14-inch-m5"

data=WebBaseLoader(url)

docs=data.load()

print(docs[0].page_content)
