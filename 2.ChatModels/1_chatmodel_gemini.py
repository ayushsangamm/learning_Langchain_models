from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 1.5,
    max_output_tokens = 68
)

response = chat_model.invoke([
    HumanMessage(content="Write a five line poem on cricket?")
])

print(response.content)