from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

AI_TEAM_SERVER = os.getenv("AI_TEAM_SERVER")
VLLM_MODEL = "gemma-4-26b"


llm = ChatOpenAI(
    model=VLLM_MODEL,
    base_url=AI_TEAM_SERVER,
    api_key="EMPTY",
    temperature=0.2,
    max_tokens=4096,
)


prompt="Hola"
ai_message = llm.invoke(prompt)
ai_message.pretty_print()
print("\n\n")
