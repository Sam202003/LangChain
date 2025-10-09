import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OpenAI API key is not set!")
    print("\nTo fix this, you have several options:")
    print("\n1. Set environment variable in terminal:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print("\n2. Create a .env file in the project root with:")
    print("   OPENAI_API_KEY=your-api-key-here")
    print("\n3. Set it directly in your IDE/terminal before running the script")
    print("\nGet your API key from: https://platform.openai.com/api-keys")
    exit(1)

# Define the embedding model
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("✅ OpenAI API key found and embeddings initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing OpenAI embeddings: {e}")
    exit(1)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "Where does Gandalf meet Frodo?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.5}, 
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")