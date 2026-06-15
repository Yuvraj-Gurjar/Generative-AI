# AI models can't "read" English. Instead, an embedding converts a word or sentence into a long list of numbers (called a vector).
# These numbers aren't random; they act as coordinates on that map. Concepts with similar meanings (like "dog" and "puppy") are assigned coordinates that place them right next to each other. Unrelated concepts (like "dog" and "skyscraper") are placed far apart.
from langchain_huggingface import HuggingFaceEmbeddings

# This will download a lightweight, popular open-source embedding model the first time you run it.
# "all-MiniLM-L6-v2" creates 384-dimensional embeddings, which are great for general use.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text = "You are going to learn Gen Ai"
vector = embeddings.embed_query(text)

print(f"The text '{text}' was converted into a vector of numbers.")
print(f"Total dimensions (length of the vector): {len(vector)}")
print(vector[:5]) # Printing just the first 5 numbers to keep the output clean