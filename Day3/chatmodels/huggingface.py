import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# 1. Load the environment variables (This reads your HF_TOKEN)
load_dotenv()

# 2. Setup the Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash", 
    task="text-generation", # Highly recommended to prevent pipeline errors
    max_new_tokens=512,     # Optional: Limits the response length
)

# 3. Wrap it in the Chat format
model = ChatHuggingFace(llm=llm)

# 4. Invoke the model
response = model.invoke("who are you?")
print(response.content)