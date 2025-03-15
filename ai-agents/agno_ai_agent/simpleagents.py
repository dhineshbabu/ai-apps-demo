# agent specific module imports
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq 
from agno.tools.duckduckgo import DuckDuckGoTools

# import specific to loading api keys for models
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Creating agents
agent = Agent(
    model=Groq(id='qwen-2.5-32b'),
    description='You are an assistant, Please reply based on the input question',
    tools=[DuckDuckGoTools()],
    markdown=True
)

# Generate response
agent.print_response("Who won the India vs Newzealand finals in CT 2025")

