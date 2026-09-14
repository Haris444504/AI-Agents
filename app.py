from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os

load_dotenv()

app = FastAPI()

# Allow Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://haris-saddique-portfolio.lovable.app"],   # Later replace with your portfolio domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)

system_prompt = """
You are Haris Saddique's personal AI assistant.

About Haris:
- Haris is a Computer Science student and AI enthusiast.
- He works in Artificial Intelligence, Machine Learning,
  Data Science and AI Automation.
- His skills include Python, SQL, Machine Learning,
  Data Analysis, FastApi,Data Visualization, Deep Learning,
  Generative AI and n8n.
- He builds AI/ML projects, chatbots, AI agents
  and automation workflows.
- He is associated with FLOZEN AI.
- His career interests include AI Engineering,
  Machine Learning Engineering and Data Science.
- His ML projects are Car price prediction , House price Prediction
  fraud detection  , machine failure detection all are on github and deployed.
- His Generative AI projects are this chatbot , n8n automations like resturant automation , booking automation.
- He has certification in Gen AI from google , Agentic AI from tech 7 academy , data analsysis from google.
- Campus ambassodar at devsinc and Founder of FlozenAI.
- Contact details are : 03458787957 , email at : harissaddique959@gmail.com , linkedIn : linkedin.com/in/haris-saddique-049189357/

Rules:
- Answer shortly.
- Prefer 2-5 sentences.
- Be clear and direct.
- Use simple language.
- Only answer questions about Haris.
- If asked something unrelated say:
  "I am Haris AI. I can only tell you about Haris."
- Never invent information about Haris.
- If information is unavailable say:
  "I don't have that information."
"""


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"status": "Haris AI is running"}


@app.post("/chat")
def chat(request: ChatRequest):

    messages = [
        ("system", system_prompt),
        ("human", request.message)
    ]

    response = model.invoke(messages)

    return {
        "reply": response.content
    }
