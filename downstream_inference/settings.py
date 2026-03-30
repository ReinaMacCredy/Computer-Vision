import os

#Thư mục gốc (IMAGES_RETRIEVAL_PROJECT)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")
IMAGES_INDEX_PATH = os.path.join(DATA_DIR, "vectorDB") # Đã khớp chữ hoa DB
MODELS_DIR = os.path.join(BASE_DIR, "models", "CLIPS")
KEY_PATH = os.path.join(BASE_DIR, "downstream_inference", "keys.json")
# Cấu hình ChromaDB
IMAGE_COLLECTION_NAME = "image_knowledge_base"
DEFAULT_K = 5