from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model = 'gemini-embedding-001', 
    dimension=32
    )

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Patna is the capital of Bihar"
]
result = embedding.embed_documents(documents)

print(str(result))