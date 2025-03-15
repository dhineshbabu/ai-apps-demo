from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Validate API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")
os.environ["GROQ_API_KEY"] = groq_api_key

# ✅ Load SentenceTransformer manually
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Define a Custom LanceDB-Compatible Embedder
class CustomEmbedder:
    def __init__(self, model):
        self.model = model
        self.dimensions = model.get_sentence_embedding_dimension()  # ✅ Required by LanceDB

    def encode(self, texts):
        """Returns embeddings as lists (LanceDB does not support raw NumPy arrays)."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

# ✅ Initialize Custom Embedder
custom_embedder = CustomEmbedder(embedding_model)

# ✅ Configure LanceDB with Custom Embedder
vector_db = LanceDb(
    uri="tmp/lancedb",
    table_name="recipes",
    search_type=SearchType.hybrid,
    embedder=custom_embedder  # ✅ Use the object, not a function
)

# ✅ Create agent
agent = Agent(
    model=Groq(id="mixtral-8x7b-32768"),
    description="You are a Thai cuisine expert!",
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results."
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=vector_db,
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

# ✅ Load knowledge base (handle errors gracefully)
try:
    agent.knowledge.load()
except Exception as e:
    print(f"Error loading knowledge base: {e}")

# Example usage
agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
agent.print_response("What is the history of Thai curry?", stream=True)
