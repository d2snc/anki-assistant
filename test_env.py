import os
from dotenv import load_dotenv

loaded = load_dotenv()
print("Loaded:", loaded)
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
