from google import genai
from google.genai import types


genai_client = genai.Client()

# Gemini Grounding

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

grounding_config = types.GenerateContentConfig(
    tools=[grounding_tool],
    temperature=0.2
)

# Get Traffic Update
def get_live_traffic(city: str) -> str:
    """
    Uses Gemini Search Grounding to infer live traffic condition.
    Returns: 'Good' | 'Medium' | 'Poor' | 'Unavailable'
    """

    prompt = f"""
Check today's traffic situation in {city} and check weather any alert based on extreme weather or traffic has been
given for the {city} .IF yes then Genrate one liner explanation.
Respond with ONLY one word:
Good, Medium, or Poor.
"""

    try:
        res = genai_client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=grounding_config
        )
        text = res.text.strip()
        return text if text in ["Good", "Medium", "Poor"] else "Unavailable"

    except Exception:
        return "Unavailable"
