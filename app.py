# Ensure your AWS credentials are configured

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_aws import BedrockEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os

# LLM Setup
model = init_chat_model("eu.anthropic.claude-3-7-sonnet-20250219-v1:0", model_provider="bedrock_converse")

# Embedding model setup
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

# QDrant Client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"),
    api_key=os.getenv("QDRANT_CLOUD_KEY") ,
)


output = model.invoke([HumanMessage(content="Hi! I'm Bob")])
print(output.content)

output = model.invoke([HumanMessage(content="Can you summarize the book 'Fairy tale' by Stephen King")])
print(output.content)