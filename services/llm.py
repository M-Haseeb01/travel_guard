import os
from groq import Groq


groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_summary(summary_data: dict) -> str:

    prompt = f"""
Travel information:

{summary_data}

You are an intelligent travel alert and guidance system.

First determine the overall travel condition.

Start the response with exactly one:

🟢 Good
🟡 Medium
🔴 Bad

Immediately place @ after the status.

Then provide exactly three short paragraphs separated by @.

Paragraph 1:
Situation – clearly explain the current conditions.

Paragraph 2:
Preparation – explain what a traveler should prepare for or be aware of.

Paragraph 3:
Recommendation – explain whether traveling is advised and why.

Do not use headings.
Do not use bullet points.
Do not use markdown.

Keep the language simple, clear and actionable.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4
    )

    return response.choices[0].message.content


def translate_text(text: str, language: str) -> str:

    prompt = f"""
Translate the following text into {language}.

Keep:
- meaning
- emojis
- formatting

Do not explain anything.

Text:
{text}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
