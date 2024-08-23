import os
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Load CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# Initialize Qdrant client
qdrant_client = QdrantClient(path="./qdrant.db")  # Local storage

# Create a Qdrant collection
collection_name = "image_vectors"
if qdrant_client.collection_exists(collection_name):
    qdrant_client.delete_collection(collection_name)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE)  # Updated with vectors_config
)

def image_to_vector(image_path: str):
    """Convert an image to a vector using CLIP."""
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).numpy().flatten()
    return image_features.tolist()

def index_images(directory: str):
    """Index all images in a directory."""
    for idx, filename in enumerate(os.listdir(directory)):
        image_path = os.path.join(directory, filename)
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            vector = image_to_vector(image_path)
            # Store in Qdrant
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=idx,  # Unique ID
                        vector=vector,
                        payload={"filename": filename}  # Store filename as metadata
                    )
                ]
            )
    print(f"Indexed {idx + 1} images.")

if __name__ == "__main__":
    # Index all images in the directory
    index_images("./data/images")
