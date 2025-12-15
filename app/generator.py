from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

class AnswerGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate(self, question: str, context: str) -> str:
        system_prompt = (
            "You are a knowledgeable and helpful medical assistant specializing in tuberculosis (TB). "
            "Your role is to provide accurate, evidence-based information about TB based on the context provided.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Carefully read and analyze ALL the context passages provided\n"
            "2. Extract relevant information from the passages to answer the question\n"
            "3. Synthesize information across multiple passages if needed\n"
            "4. If the passages contain partial information, provide what's available\n"
            "5. Only say information is 'not available' if the passages truly contain nothing relevant\n"
            "6. For greetings, respond warmly and offer to help with TB questions\n"
            "7. If the question is completely unrelated to TB, politely redirect to TB topics\n"
            "8. Cite specific details from the passages when possible\n"
            "9. Keep responses clear, concise, and patient-friendly"
        )

        user_prompt = f"""
CONTEXT PASSAGES:
{context}

USER QUESTION:
{question}

TASK:
Analyze the context passages carefully and provide a helpful answer to the user's question.
Look for both direct and indirect information that relates to the question.
If you find relevant information, use it to construct a clear and accurate answer.
"""

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Slightly higher for more natural responses
            top_p=0.95,
            max_tokens=600  # Increased for more complete answers
        )

        return response.choices[0].message.content.strip()