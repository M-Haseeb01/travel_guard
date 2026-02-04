from langchain_core.messages import HumanMessage
from config import GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.4
) 

def generate_summary(summary_data: dict) -> str:
    prompt = f"""
    {summary_data}
    
    You are an intelligent travel alert and guidance system.
    
    First, determine the overall travel condition and start the response with one of the following:
    🟢 Good | 🟡 Medium | 🔴 Bad  
    Immediately place a @ symbol after this status.
    
    Then provide **three short and concise paragraphs**, separated by a @ symbol (no headings, no bullet points, no markdown).
    
    The paragraphs should be in this exact order:
    1) Situation – clearly explain the current conditions.
    2) Preparation – what a traveler should prepare or be aware of.
    3) Recommendation – whether traveling is advised or not, and why.
    
    Keep the language simple, clear, and actionable.
    """

   
    return llm.invoke([HumanMessage(content=prompt)]).text


def translate_text(text: str, language: str) -> str:
    """Translate text using LLM."""
    prompt = f"Translate the following text to {language} keeping meaning and emojis. Do not explain:\n{text}"
    return llm.invoke([HumanMessage(content=prompt)]).text
