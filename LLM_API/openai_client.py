import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "API_KEY"
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
client = OpenAI()
print("Success")