import os
from dotenv import load_dotenv

#Thư mục gốc (IMAGES_RETRIEVAL_PROJECT)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")
IMAGES_INDEX_PATH = os.path.join(DATA_DIR, "vectorDB") # Đã khớp chữ hoa DB
MODELS_DIR = os.path.join(BASE_DIR, "models", "CLIPS")
# Cấu hình ChromaDB
IMAGE_COLLECTION_NAME = "image_knowledge_base"
DEFAULT_K = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "")
CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "")