import os
import json
from sentence_transformers import SentenceTransformer
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec
from .config import settings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# Initialize Pinecone and embedding model
pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.index_name)
model = SentenceTransformer(settings.model_name)
#os.environ["GROQ_API_KEY"] = "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"

# Initialize Groq LLM
groq_llm = ChatGroq(temperature=0, api_key=api_key, model_name="mixtral-8x7b-32768")

# Define the prompt template for explanation generation
prompt_template = PromptTemplate(
    template="Explain why this resume is a good fit for the query '{query}':\n{resume_text} in 15 words",
    input_variables=["query", "resume_text"]
)

# Create the LLMChain for explanation generation
llm_chain = LLMChain(prompt=prompt_template, llm=groq_llm)

def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '_')
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '_')
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def sanitize_metadata(metadata):
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            sanitized.update({f"{key}_{k}": v for k, v in sanitize_metadata(value).items()})
        elif isinstance(value, list):
            sanitized[key] = [str(i) for i in value] if not all(isinstance(i, str) for i in value) else value
        else:
            sanitized[key] = str(value) if not isinstance(value, (str, int, float, bool)) else value
    return sanitized

def perform_rag_pipeline(query: str):
    # Step 1: Perform Semantic Search
    query_embedding = model.encode([query])[0]
    search_results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)

    explanations = []
    for match in search_results['matches']:
        resume_text = " ".join([f"{k}: {v}" for k, v in match['metadata'].items()])
        prompt_input = {
            "query": query,
            "resume_text": resume_text
        }
        # Generate the explanation using the Groq model
        response = llm_chain.invoke(prompt_input)
        explanation = response['text'].strip() if 'text' in response else "No response text found"
        
        explanations.append({
            "score": match['score'],
            "resume": match['metadata'],
            "explanation": explanation
        })

    return explanations
