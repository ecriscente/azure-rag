import os
# import openai
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import AzureSearch
import pandas as pd
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


app = FastAPI()

# openai.api_base = os.getenv("OPENAI_API_BASE")  # Your Azure OpenAI resource's endpoint value.
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_type = "azure"
# openai.api_version = "2023-05-15" 

# embeddings = OpenAIEmbeddings(deployment="demo-embedding", chunk_size=1)

# Connect to Azure Cognitive Search
# acs = AzureSearch(azure_search_endpoint=os.getenv('SEARCH_SERVICE_NAME'),
#                  azure_search_key=os.getenv('SEARCH_API_KEY'),
#                  index_name=os.getenv('SEARCH_INDEX_NAME'),
#                  embedding_function=embeddings.embed_query)

df = pd.read_csv('../wine-ratings.csv')
df = df[df['variety'].notna()] # remove any NaN values as it blows up serialization
data = df.sample(700).to_dict('records') # Get only 700 records. More records will make it slower to index
len(data)

encoder = SentenceTransformer('all-MiniLM-L6-v2') # Model to create embeddings

# create the vector database client
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance

# Create collection to store wines
qdrant.recreate_collection(
    collection_name="top_wines",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

# vectorize!
qdrant.upload_points(
    collection_name="top_wines",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["notes"]).tolist(),
            payload=doc,
        ) for idx, doc in enumerate(data) # data is the variable holding all the wines
    ]
)

class Body(BaseModel):
    query: str


@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')
def ask(body: Body):
    """
    Use the query parameter to interact with the Azure OpenAI Service
    using the Azure Cognitive Search API for Retrieval Augmented Generation.
    """
    # search_result = search(body.query)
    # chat_bot_response = assistant(body.query, search_result)
    chat_bot_response = assistant(body.query)
    return {'response': chat_bot_response}



def search(query):
    """
    Send the query to Azure Cognitive Search and return the top result
    """
    # docs = acs.similarity_search_with_relevance_scores(
    #     query=query,
    #     k=5,
    # )

    hits = qdrant.search(
        collection_name="top_wines",
        query_vector=encoder.encode(query).tolist(),
        limit=3
    )
    # result = docs[0][0].page_content
    # define a variable to hold the search results
    search_results = [hit.payload for hit in hits]
    print(search_results)
    return search_results


# def assistant(query, context):
def assistant(query):
    messages=[
        # Set the system characteristics for this chat bot
        {"role": "system", "content": "You are just a normal chatbot."},

        # Set the query so that the chatbot can respond to it
        {"role": "user", "content": query},

        # Add the context from the vector search results so that the chatbot can use
        # it as part of the response for an augmented context
        # {"role": "assistant", "content": str(context)}
    ]

    # response = openai.ChatCompletion.create(
    #     engine="demo-alfredo",
    #     messages=messages,
    # )
    client = OpenAI(
        base_url="http://127.0.0.1:8080/v1", # "http://<Your api-server IP>:port"
        api_key = "sk-no-key-required"
    )
    response = client.chat.completions.create(
        model="LLaMA_CPP",
        messages=messages
    )
    print(response)
    final_response = response.choices[0].message.content
    print(final_response)
    return final_response