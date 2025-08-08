# llm_handler.py
import os
import anthropic

# --- Initialize the Anthropic Client ---
# The client automatically picks up the ANTHROPIC_API_KEY from environment variables
try:
    client = anthropic.Anthropic()
except Exception as e:
    print(f"Failed to initialize Anthropic client: {e}")
    client = None

def get_answer_from_llm(context: str, question: str) -> str:
    """
    Uses an Anthropic Claude model to generate an answer based on the provided context and question.
    """
    if not client:
        return "Anthropic client is not initialized. Please check your API key."

    # System prompts are great for telling the model how to behave
    system_prompt = """
    You are an expert Q&A system. Your task is to answer the user's question based *only* on the provided context.
    If the answer is not found in the context, state that clearly by responding with "I could not find an answer in the provided document." Do not make up information.
    """

    # The user message contains the specific data for the task
    user_message = f"""
    Context:
    ---
    {context}
    ---

    Question: {question}
    """

    try:
        message = client.messages.create(
            # We recommend claude-3-5-sonnet, as it's fast, affordable, and highly intelligent.
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            temperature=0.0, # We want factual, deterministic answers
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )
        # The response text is in the first content block
        return message.content[0].text.strip()

    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return "There was an error generating the answer from Anthropic."