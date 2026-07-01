from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os

import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model = 'gemini-embedding-001',
    google_api_key = os.getenv("GEMINI_API_KEY")
)

documents = [
    "Virat kohli is an Indian Cricketer known for his aggressive batting and leadership",
    "Ms Dhoni is a former indian captain famous for his calm demeanor and finishing skills",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries",
    "Jasprit Bumrah is an indian fast bowler known for his unorthodox action and yorkers"
]

query = 'tell me about virat Kohli'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0] #dono ko 2d vector me bhjna hai sbse important step hai ye

best_idx = np.argmax(similarities)

print(f"Best match : {documents[best_idx]} (score : {similarities[best_idx] : 4f})")

