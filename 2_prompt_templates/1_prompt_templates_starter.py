from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max."

prompt_template = ChatPromptTemplate.from_template(template)

# Format the prompt with variables
formatted_prompt = prompt_template.invoke({"tone": "friendly", "company": "Meta", "position": "Software Engineer", "skill": "Python"})

# Send the formatted prompt to the LLM
result = llm.invoke(formatted_prompt)

print(result.content)

messages = [
    ("system","Yor are a comedian who tells jokes about {topic}"),
    ("human","Tell me a {jokecount} jokes"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "AI", "jokecount": 3})

result = llm.invoke(prompt)
print(result.content)



