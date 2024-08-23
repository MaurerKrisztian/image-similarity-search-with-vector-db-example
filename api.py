from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance
from PIL import Image
import torch
import os
import io

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# Initialize Qdrant client
qdrant_client = QdrantClient(path="./qdrant.db")  # Local storage
collection_name = "image_vectors"

def image_to_vector(image: Image.Image):
    """Convert an image to a vector using CLIP."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).numpy().flatten()
    return image_features.tolist()

@app.get("/preloaded-images/")
async def get_preloaded_images():
    """Return a list of filenames for preloaded images."""
    image_directory = "./data/images"
    images = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return images

@app.get("/search-image-by-filename/")
async def search_image_by_filename(
    filename: str,
    top_k: int = 5,
    distance: str = "cosine"
):
    """Find similar images based on a given image filename."""
    image_path = os.path.join("./data/images", filename)
    if not os.path.exists(image_path):
        return {"error": "Image not found"}

    image = Image.open(image_path)
    query_vector = image_to_vector(image)

    # Set the distance metric based on the setting
    if distance == "cosine":
        metric = Distance.COSINE
    elif distance == "euclidean":
        metric = Distance.EUCLID
    else:
        metric = Distance.DOT

    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )

    similar_images = [result.payload["filename"] for result in results]
    return similar_images

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
