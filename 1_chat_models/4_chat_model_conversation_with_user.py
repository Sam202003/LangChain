from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

chat_history = []

system_message = SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks.")
chat_history.append(system_message)

while True:

    query= input("You:")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    
    print(f"AI: {response}")

   
print("___Message History___")
print(chat_history)