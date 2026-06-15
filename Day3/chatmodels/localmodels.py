from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# 1. Initialize the local pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        # max_new_tokens=512,  
    )
)

# 2. Wrap it in the Chat format
chat_model = ChatHuggingFace(llm=llm)

# 3. Invoke the model
print("Loading model and generating response...")
result = chat_model.invoke("what is full stack development?")

# 4. Print the clean text response
print("\nResponse:")
print(result.content)