# System Message: Defines the AI's role and sets the context for the conversation 

# For example, the system message might be:"You are a maketing expert."

# Human Message: The user's input or query

# AI Message: The AI's response to the user's input

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

messages=[
    SystemMessage("You are an expert in social media content strategy."),
    HumanMessage("Give a short tip to create engaging posts on Instagram."),
    AIMessage("Use emojis and keep posts concise and to the point. Use hashtags and tags to reach a wider audience.")
]
result = llm.invoke(messages)
print(result);