from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForMaskedLM, AutoTokenizer
from openai import OpenAI
import torch
import pandas as pd
from pinecone import Pinecone

app = FastAPI()

# Initialize models
model_id = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_sparse = AutoModelForMaskedLM.from_pretrained(model_id)

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")
index = pc.Index("recipe-project")

recipes = pd.read_pickle("recipes_with_vectors.pkl")

class QueryRequest(BaseModel):
    user_query: str
    recipe_type: str

def to_dense_vector_openAI(text, client, model, dimensions):
    dense_vectors = client.embeddings.create(model=model, dimensions=dimensions, input=[text])
    return dense_vectors.data[0].embedding

def to_sparse_vector(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt')
    output = model(**tokens)
    vec = torch.max(torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1)[0].squeeze()
    cols = vec.nonzero().squeeze().cpu().tolist()
    weights = vec[cols].cpu().tolist()
    return dict(zip(cols, weights))

def hybrid_search(sparse_dict, dense_vectors, alpha):
    hsparse = {
        "indices": list(sparse_dict.keys()),
        "values": [v * (1 - alpha) for v in list(sparse_dict.values())]
    }
    hdense = [v * alpha for v in dense_vectors]
    return hdense, hsparse

def answer_query(user_query, recipe_type):
    text_dense_vector = to_dense_vector_openAI(user_query, client, "text-embedding-3-small", 768)
    text_sparse_vector = to_sparse_vector(user_query, tokenizer, model_sparse)
    dense_vector, sparse_dict = hybrid_search(text_sparse_vector, text_dense_vector, alpha=0.5)

    retrieved_items = index.query(
        vector=dense_vector,
        sparse_vector=sparse_dict,
        include_values=False,
        include_metadata=True,
        top_k=3,
        filter={"recipe_type": {"$eq": recipe_type}}
    )

    retrieved_ids = [item.get("metadata").get("ID") for item in retrieved_items.get("matches")]
    return [x for x in recipes[recipes.ID.isin(retrieved_ids)].output.values]

@app.post("/get_recipe/")
def get_recipe(request: QueryRequest):
    recipes = answer_query(request.user_query, request.recipe_type.lower())
    return {"recipes": recipes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
