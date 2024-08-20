from fastapi import FastAPI, Query
from .services import router as search_router
import json
from .rag import perform_rag_pipeline

app = FastAPI()

app.include_router(search_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Pipeline API"}

@app.get("/generate-explanations")
def generate_explanations(query: str = Query(..., description="The search query to generate explanations for")):
    # Perform the RAG pipeline to generate explanations
    explanations = perform_rag_pipeline(query)

    # Convert the list of dictionaries to a JSON string with pretty printing
    json_data = json.dumps(explanations, indent=4)

    # Write the JSON string to a file named "output.json"
    with open("output.json", "w") as json_file:
        json_file.write(json_data)

    return {"message": "Explanations generated and 'output.json' created successfully.", "explanations": explanations}
