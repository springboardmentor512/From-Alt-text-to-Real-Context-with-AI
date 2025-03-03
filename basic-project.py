import os
import dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL") # gemini-1.5-flash
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # A3zaSyA2ljk9PrtRZ-MJswTLFWdbsl5G2O0M7Eo

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

TEMPLATE = """
You are an AI Medical Assistant trained to provide accurate medical information and guidance. You will:
1. Answer medical questions with scientific accuracy
2. Provide general health information and wellness advice
3. Help interpret common medical terminology
4. Suggest when to seek professional medical care

You are only allowed to answer questions about the medical field.

IMPORTANT: I am an AI assistant and cannot diagnose conditions or replace professional medical advice. For any serious medical concerns, please consult a qualified healthcare provider.

{input}
"""

prompt = PromptTemplate.from_template(TEMPLATE)

chain = prompt | llm

user_input = input("Enter a question: ")

response = chain.invoke({"input": user_input})

print(response.content)