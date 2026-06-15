from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage

# 1. Load the environment variables (.env file)
load_dotenv()

# 2. Use Groq instead of OpenAI
model = init_chat_model(
    # if u want some creative task temperature will be high otherwise low range(0-1)
    model="llama-3.3-70b-versatile",temperature=0.9,max_tokens=25, 
    model_provider="groq"
)

# to save the history of your chats
print("choose your Ai mode")
print("press 1 for Angry mode")
print("press 2 for funny mode")
print("press 3 for sad mode")
choice=int(input("Tell your response:-"))

if choice==1:
    mode="You are a angry ai agent.Response agressively."
elif choice==2:
    mode="you are funny ai agent answer accondingly"
elif choice==3:
    mode="You are sad ai agent answer aressively like that"
messages=[
    SystemMessage(content=mode)
]

print("-------Welcome type 0 to exit the application-------")
while True:
    prompt=input("You:")
    messages.append(HumanMessage(content=prompt))
    if prompt=="0":
        break

# 3. Invoke the model
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))

# 4. Print the clean text response
    print("Bot:",response.content)
print(messages)