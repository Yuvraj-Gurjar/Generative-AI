from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 1. Load the environment variables (.env file)
load_dotenv()

# 2. Use Groq instead of OpenAI
model = init_chat_model(
    # if u want some creative task temperature will be high otherwise low range(0-1)
    model="llama-3.3-70b-versatile",temperature=0.9,max_tokens=25, 
    model_provider="groq"
)

# 3. Invoke the model
response = model.invoke("write a poem on AI in hinglish")

# 4. Print the clean text response
print(response.content)