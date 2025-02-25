import streamlit as st
from openai import OpenAI
import re
import json
import spacy
import torch
import os
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pinecone import Pinecone, ServerlessSpec

def hybride_search(sparse_dict, dense_vectors, alpha):

    # check alpha value is in range
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        "indices": list(sparse_dict.keys()),
        "values": [v * (1 - alpha) for v in list(sparse_dict.values())]
    }
    hdense = [v * alpha for v in dense_vectors]
    return hdense, hsparse

def answer_query(user_query, recipe_type):
    model = "text-embedding-3-small"
    model_id = "naver/splade-cocondenser-ensembledistil"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_sparse = AutoModelForMaskedLM.from_pretrained(model_id)


    client = OpenAI(
        api_key= os.getenv("OPENAI_API_KEY")
    )

    pc = Pinecone(api_key="pcsk_3ACEUQ_Bb4rT8NN1EAzY1zamEfBGmvZME68mEN3tEPZPNguUqcguojhucB8GZL2cPBFeLS")
    recipes = pd.read_pickle("recipes_with_vectors.pkl")
    index = pc.Index("recipe-project")

    text_dense_vector = to_dense_vector_openAI(user_query, client, model, 768)
    text_sparse_vector = to_sparse_vector(user_query, tokenizer, model_sparse)

    dense_vector, sparse_dict = hybride_search(text_sparse_vector, text_dense_vector, alpha=0.5)

    retrieved_items = index.query(vector=dense_vector,
                                  sparse_vector=sparse_dict,
                                  include_values=False,
                                  include_metadata=True,
                                  top_k=3,
                                  filter={"recipe_type": {"$eq": recipe_type}})

    #print("retrieved_items ",retrieved_items)

    retrieved_ids = [item.get("metadata").get("ID") for item in retrieved_items.get("matches")]

    return [x for x in recipes[recipes.ID.isin(retrieved_ids)].output.values]

def to_dense_vector_openAI(text, client, model, dimensions):
    dense_vectors = client.embeddings.create(model=model, dimensions=dimensions, input=[text])
    return dense_vectors.data[0].embedding


def to_sparse_vector(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt')
    output = model(**tokens)
    vec = torch.max(
        torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1
    )[0].squeeze()

    cols = vec.nonzero().squeeze().cpu().tolist()
    weights = vec[cols].cpu().tolist()
    sparse_dict = dict(zip(cols, weights))
    return sparse_dict

def format_recipe_output(recipe):
    title = recipe['title']
    ingredients = recipe['ingredients']
    direction = recipe['direction']

    formatted_output = f"🍽️ *Recipe Suggestion: {title}*\n\n"
    formatted_output += "🔪 *Ingredients:*\n" + ingredients.replace('- ', '• ') + "\n\n"
    formatted_output += "📜 *Directions:*\n" + direction + "\n"

    return formatted_output

def main():
    st.title("🍴Recipe Recommender Chatbot")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("API key is missing. Make sure to set OPENAI_API_KEY.")

    print(f"Using API key: {api_key[:5]}... (hidden for security)")

    # Input field for the user query
    user_query = st.text_input("Hi, I am your recipe bot, how can I help you?","", placeholder="Eg. I want to cook an italian dish with meat")

    # Dropdown for dietary preferences
    recipe_type = st.selectbox(
        "Select your dietary preference:",
        ["Regular", "Vegetarian", "Vegan"],
    )

    if st.button("Submit"):
      if user_query:
          # Fetch the recipe suggestions (assuming it's a list of recipes)
          recommendation_response = answer_query(user_query, recipe_type.lower())  # Example: returns a list of 3 recipes

          # Ensure you have multiple recipes to display
          if isinstance(recommendation_response, list) and len(recommendation_response) >= 3:
              # Divide the screen into three columns
              col1, col2, col3 = st.columns(3)

              # Format and display recipes in each column
              with col1:
                  st.success(format_recipe_output(recommendation_response[0]))
              with col2:
                  st.success(format_recipe_output(recommendation_response[1]))
              with col3:
                  st.success(format_recipe_output(recommendation_response[2]))
          else:
              st.warning("Could not fetch three recipes. Please try again.")
      else:
          st.warning("Please enter a query to get a recipe suggestion.")

if __name__ == "__main__":
    main()
