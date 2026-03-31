import os
import json
import httpx
import base64
import mimetypes
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from PIL import Image
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from openai import OpenAI
from openai import APIStatusError
import io
import math
import re

# Import cấu hình từ file settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from downstream_inference.settings import (
    MODELS_DIR, IMAGES_DIR, IMAGES_INDEX_PATH,
    IMAGE_COLLECTION_NAME, DEFAULT_K, OPENAI_API_KEY, OPENAI_BASE_URL,
    OPENAI_TEXT_MODEL_ID, OPENAI_VISION_MODEL_ID, CLIP_MODEL_ID
)


class ImageRAG:
    def __init__(self):
        self.openai_api_key, self.openai_base_url = self._load_api_config()

        self.http_client = httpx.Client(verify=False, timeout=120.0)

        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            http_client=self.http_client
        )

        self.image_loader = None
        self.clip_model = None
        self.chroma_client = None
        self.collection = None

        # QUAN TRỌNG:
        # similarity = 1 - distance chỉ đúng khi metric là cosine
        self.distance_metric = "cosine"

        self._init_resources()

    def _load_api_config(self):
        """Load API key and base URL from environment variables."""
        api_key = OPENAI_API_KEY or os.getenv("GOOGLE_API_KEY", "")
        base_url = OPENAI_BASE_URL
        return api_key, base_url

    def _init_resources(self):
        """Initialize Image RAG resources (ImageLoader + CLIP + ChromaDB)"""
        print("\n>>> [INIT] Image RAG Engine...")

        if not CLIP_MODEL_ID:
            raise ValueError("Missing CLIP_MODEL_ID in environment configuration.")

        self.image_loader = ImageLoader()
        local_ckpt_path = os.path.join(MODELS_DIR, "open_clip_pytorch_model.bin")

        try:
            if os.path.exists(local_ckpt_path):
                print(f"   + Loading CLIP model from local checkpoint: {local_ckpt_path}")
                self.clip_model = OpenCLIPEmbeddingFunction(
                    model_name=CLIP_MODEL_ID,
                    checkpoint=local_ckpt_path
                )
                print("✅ CLIP model loaded successfully from local checkpoint.")
            else:
                raise FileNotFoundError(f"Local CLIP checkpoint not found at {local_ckpt_path}")

        except Exception as e:
            print(f"⚠️ Failed to load local CLIP model: {e}")
            print("   -> Falling back to HuggingFace model download")
            self.clip_model = OpenCLIPEmbeddingFunction(model_name=CLIP_MODEL_ID)
            print("✅ CLIP model loaded from HuggingFace.")

        try:
            print(f"   + Connecting to ChromaDB at {IMAGES_INDEX_PATH}...")
            self.chroma_client = chromadb.PersistentClient(path=IMAGES_INDEX_PATH)

            # SỬA QUAN TRỌNG:
            # đặt metadata hnsw:space để xác định metric một cách rõ ràng
            self.collection = self.chroma_client.get_or_create_collection(
                name=IMAGE_COLLECTION_NAME,
                embedding_function=self.clip_model,
                data_loader=self.image_loader,
                metadata={"hnsw:space": "cosine"}
            )

            self.distance_metric = self._get_collection_distance_metric()
            print(f"✅ Image ChromaDB connected successfully. Distance metric = {self.distance_metric}")

        except Exception as e:
            print(f"❌ Error while connecting to ChromaDB: {e}")
            raise

    def _get_collection_distance_metric(self) -> str:
        """
        Lấy metric từ metadata collection.
        Mặc định coi là cosine nếu không đọc được, nhưng sẽ cảnh báo.
        """
        try:
            meta = self.collection.metadata or {}
            metric = meta.get("hnsw:space", "cosine")
            return metric
        except Exception:
            print("⚠️ Cannot read collection metric metadata. Fallback to 'cosine'.")
            return "cosine"

    def _distance_to_similarity(self, distance: float) -> Optional[float]:
        """
        Chuyển distance -> similarity nếu và chỉ nếu metric hỗ trợ diễn giải rõ ràng.
        - cosine distance: similarity = 1 - distance
        - l2 / ip: không nên cưỡng ép đổi theo công thức trên
        """
        if distance is None:
            return None

        if self.distance_metric == "cosine":
            return 1.0 - float(distance)

        # Với l2 hoặc ip, tránh diễn giải sai
        return None

    def date_to_int(self, date_str: str) -> int:
        try:
            return int(date_str.replace("-", ""))
        except Exception:
            return int(datetime.now().strftime("%Y%m%d"))

    def list_date_to_int(self, date_list: list) -> list:
        if not date_list:
            return []
        return [self.date_to_int(d) for d in date_list]

    # ==========================================
    # LLM / INTENT
    # ==========================================
    def _call_llm_with_retry(self, model_name, messages, max_tokens=800, temperature=0.3, max_retries=3):
        """Gọi LLM API với retry khi gặp lỗi 502/503/timeout"""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                print(f"   -> Lần thử {attempt}/{max_retries}...")
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response
            except Exception as e:
                last_error = e
                error_msg = str(e)
                is_retryable = any(code in error_msg for code in ["502", "503", "504", "timeout", "Service unavailable", "Bad Gateway"])
                if is_retryable and attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"   ⚠️ Lỗi server (lần {attempt}): {error_msg[:100]}... Đợi {wait_time}s rồi thử lại.")
                    time.sleep(wait_time)
                else:
                    raise last_error

    def _extract_text_content(self, response) -> str:
        raw_text = response.choices[0].message.content
        if isinstance(raw_text, list):
            raw_text = "".join(
                item.get("text", "")
                for item in raw_text
                if isinstance(item, dict)
            )
        return (raw_text or "").strip()

    def _call_fireworks_vision_with_retry(
        self,
        model_name,
        prompt,
        image_urls,
        max_tokens=800,
        temperature=0.1,
        max_retries=3
    ):
        """Gọi Fireworks vision theo định dạng completions + extra_body.images."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                print(f"   -> Vision attempt {attempt}/{max_retries}...")
                response = self.client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"images": image_urls}
                )
                return response
            except Exception as e:
                last_error = e
                error_msg = str(e)
                is_retryable = any(code in error_msg for code in ["502", "503", "504", "timeout", "Service unavailable", "Bad Gateway"])
                if is_retryable and attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"   ⚠️ Vision server error (attempt {attempt}): {error_msg[:100]}... Wait {wait_time}s.")
                    time.sleep(wait_time)
                else:
                    raise last_error

    def ping_provider(self) -> Dict[str, Any]:
        """Ping nhà cung cấp OpenAI-compatible và trả về trạng thái chi tiết."""
        if not self.openai_base_url:
            return {
                "ok": False,
                "status_code": None,
                "message": "Thiếu OPENAI_BASE_URL."
            }

        if not self.openai_api_key:
            return {
                "ok": False,
                "status_code": None,
                "message": "Thiếu OPENAI_API_KEY."
            }

        if not OPENAI_TEXT_MODEL_ID and not OPENAI_VISION_MODEL_ID:
            return {
                "ok": False,
                "status_code": None,
                "message": "Thiếu OPENAI_TEXT_MODEL_ID hoặc OPENAI_VISION_MODEL_ID."
            }

        checks = []
        for label, model_id in [("text", OPENAI_TEXT_MODEL_ID), ("vision", OPENAI_VISION_MODEL_ID)]:
            if not model_id:
                continue
            try:
                self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0
                )
                checks.append(f"{label}: ok (`{model_id}`)")
            except APIStatusError as e:
                message = str(e)
                if e.status_code == 401:
                    message = f"{label}: 401 Unauthorized - API key không hợp lệ hoặc thiếu quyền."
                elif e.status_code == 404:
                    message = f"{label}: 404 Not Found - endpoint hoặc model ID `{model_id}` không đúng."
                elif e.status_code == 400:
                    message = f"{label}: 400 Bad Request - model ID `{model_id}` hoặc request format không phù hợp."
                return {
                    "ok": False,
                    "status_code": e.status_code,
                    "message": message
                }
            except Exception as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                return {
                    "ok": False,
                    "status_code": status_code,
                    "message": f"{label}: {str(e)}"
                }

        return {
            "ok": True,
            "status_code": 200,
            "message": " | ".join(checks)
        }

    def extract_query_intent(self, user_query: str, model_name=OPENAI_TEXT_MODEL_ID):
        """Bóc tách ý định người dùng thành JSON chứa từ khóa tiếng Anh và ngày tháng"""
        if not model_name:
            raise ValueError("Missing OPENAI_TEXT_MODEL_ID in environment configuration.")
        print(f"\n🕵️ Đang phân tích yêu cầu: '{user_query}'")

        prompt = f"""
        Bạn là hệ thống trích xuất dữ liệu. Hôm nay là năm 2026.
        Hãy phân tích yêu cầu: "{user_query}"

        Chỉ được trả về DUY NHẤT một JSON object hợp lệ, không markdown, không giải thích thêm, không code fence.
        JSON phải có các trường:
        - "english_query": Dịch chủ đề chính cần tìm trong ảnh sang tiếng Anh (ngắn gọn, VD: "yellow dog").
        - "start_date": Ngày bắt đầu (YYYY-MM-DD). VD: "năm nay" là "2026-01-01". Nếu không có thì để null.
        - "end_date": Ngày kết thúc (YYYY-MM-DD). VD: "năm nay" là "2026-12-31". Nếu không có thì để null.
        - "max_results": Số lượng ảnh muốn tìm (số nguyên). Mặc định 4.
        """

        try:
            response = self._call_llm_with_retry(
                model_name=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0
            )
            raw_text = self._extract_text_content(response)

            if not raw_text:
                raise ValueError("Provider responded with empty content.")

            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()

            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(0).strip()

            intent_data = json.loads(raw_text)
            print(f"✅ Bóc tách thành công: {json.dumps(intent_data, indent=2, ensure_ascii=False)}")
            return intent_data
        except Exception as e:
            error_msg = str(e)
            short_error = error_msg[:200] if len(error_msg) > 200 else error_msg
            print(f"⚠️ Lỗi bóc tách ý định: {short_error}")
            return {
                "english_query": user_query,
                "start_date": None,
                "end_date": None,
                "max_results": 4,
                "_llm_error": f"Provider đã phản hồi nhưng nội dung không phải JSON hợp lệ: {short_error}",
                "_llm_raw_output": raw_text if 'raw_text' in locals() else ""
            }

    # ==========================================
    # RETRIEVAL
    # ==========================================
    def retrieve_images(
        self,
        user_query: str,
        date: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results=DEFAULT_K
    ) -> List[dict]:
        """Hàm truy xuất thuần Python trả về list dict chứa thông tin ảnh"""
        print("\n>>> [Image RAG] Retrieving images for query:", user_query)
        print("    With specific date:", date)
        print("    With period from", start_date, "to", end_date, "\n")

        start = time.time()
        images_data = self.image_uris(
            query_text=user_query,
            date=date,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            max_distance=None
        )
        print("    Retrieved images in:", time.time() - start, "seconds")

        if not images_data:
            print(f"⚠️ Didn't find any images for: '{user_query}'")
            return []

        for item in images_data:
            uri = item["uri"]
            if not os.path.isabs(uri):
                uri = os.path.join(os.path.dirname(__file__), uri)
            if not os.path.exists(uri):
                raise FileNotFoundError(f"Image file doesn't exist: {uri}")
            item["uri"] = uri

        return images_data

    def image_uris(
        self,
        query_text,
        date: list = None,
        start_date: str = None,
        end_date: str = None,
        max_distance=None,
        max_results=DEFAULT_K
    ):
        if date is not None:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=max_results,
                where={"date": {"$in": self.list_date_to_int(date)}},
                include=['uris', 'distances', 'metadatas']
            )
        elif start_date is not None and end_date is not None:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=max_results,
                where={
                    "$and": [
                        {"date": {"$gte": self.date_to_int(start_date)}},
                        {"date": {"$lte": self.date_to_int(end_date)}}
                    ]
                },
                include=['uris', 'distances', 'metadatas']
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=max_results,
                include=['uris', 'distances', 'metadatas']
            )

        filtered_results = []
        result_uris = results.get('uris', [[]])[0]
        result_distances = results.get('distances', [[]])[0]
        result_metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else [{} for _ in result_uris]

        for rank, (uri, distance, metadata) in enumerate(zip(result_uris, result_distances, result_metadatas), start=1):
            distance = float(distance)

            if max_distance is not None and distance > max_distance:
                continue

            similarity = self._distance_to_similarity(distance)

            filtered_results.append({
                "rank": rank,
                "uri": uri,
                "distance": distance,
                "similarity": similarity,  # chỉ có giá trị khi metric = cosine
                "metric": self.distance_metric,
                "metadata": metadata or {}
            })

        return filtered_results

    def image_to_image_retrieval(self, query_image_path: str, max_results=DEFAULT_K, max_distance=None):
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh truy vấn: {query_image_path}")

        image = Image.open(query_image_path).convert("RGB")
        image_array = np.array(image)

        results = self.collection.query(
            query_images=[image_array],
            n_results=max_results,
            include=['uris', 'distances', 'metadatas']
        )

        filtered_results = []
        result_uris = results.get('uris', [[]])[0]
        result_distances = results.get('distances', [[]])[0]
        result_metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else [{} for _ in result_uris]

        for rank, (uri, distance, metadata) in enumerate(zip(result_uris, result_distances, result_metadatas), start=1):
            distance = float(distance)

            if max_distance is not None and distance > max_distance:
                continue

            similarity = self._distance_to_similarity(distance)

            filtered_results.append({
                "rank": rank,
                "uri": uri,
                "distance": distance,
                "similarity": similarity,
                "metric": self.distance_metric,
                "metadata": metadata or {}
            })

        return filtered_results

    # ==========================================
    # GENERATION (RAG)
    # ==========================================
    def _encode_image_to_base64(self, image_path, max_size=(512, 512)):
        """
        Resize và nén ảnh trước khi chuyển sang Base64 để tránh lỗi payload lớn
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img.thumbnail(max_size, Image.Resampling.LANCZOS)

                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=70)

                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"⚠️ Lỗi khi nén ảnh {image_path}: {e}")
            return ""

    def rag_generate_explanation(self, query_text, retrieved_data, model_name=OPENAI_VISION_MODEL_ID):
        if not retrieved_data:
            return "Không tìm thấy hình ảnh nào phù hợp trong cơ sở dữ liệu để phân tích."
        if not model_name:
            raise ValueError("Missing OPENAI_MODEL_ID in environment configuration.")

        retrieved_uris = [item["uri"] for item in retrieved_data]

        print(f"🧠 Đang gửi {len(retrieved_uris)} ảnh lên LLM để phân tích...")

        metadata_context = ""
        for i, item in enumerate(retrieved_data):
            uri = item["uri"]
            file_name = os.path.basename(uri)

            metadata = item.get("metadata", {}) or {}
            date_raw = metadata.get("date", "Không rõ")

            date_str = "Không rõ"
            if isinstance(date_raw, int):
                date_int = str(date_raw)
                if len(date_int) == 8:
                    date_str = f"{date_int[:4]}-{date_int[4:6]}-{date_int[6:]}"
                else:
                    date_str = date_int
            else:
                date_str = str(date_raw)

            similarity_str = (
                f"{item['similarity']:.4f}" if item.get("similarity") is not None else "N/A"
            )

            metadata_context += (
                f"- Ảnh {i+1} (Tên file: {file_name}): "
                f"Ngày chụp/lưu = {date_str}, "
                f"distance = {item['distance']:.4f}, "
                f"similarity = {similarity_str}, "
                f"metric = {item.get('metric', 'unknown')}\n"
            )

        system_prompt = f"""Bạn là hệ thống AI giải thích kết quả truy xuất ảnh.

Mục tiêu của bạn là cung cấp cho người dùng phần giải thích NGẮN GỌN, CHÍNH XÁC, TẬP TRUNG và DỄ HIỂU.

Bắt buộc tuân thủ:
- Chỉ trả lời tối đa 4 câu ngắn hoặc 4 gạch đầu dòng ngắn.
- Câu đầu tiên phải đưa ra kết luận chung về mức độ khớp giữa kết quả và yêu cầu.
- Chỉ nêu những ý quan trọng nhất.
- Không mô tả dài dòng từng ảnh nếu không cần thiết.
- Không lặp lại nguyên văn yêu cầu của người dùng.
- Không suy đoán vượt quá những gì ảnh hoặc metadata cho phép kết luận.
- Nếu chưa đủ chắc chắn, phải nói rõ mức độ chắc chắn còn hạn chế.
- Nếu có ngày tháng, chỉ nhắc khi nó thực sự liên quan đến truy vấn.
- Ưu tiên diễn đạt theo kiểu: kết luận trước, lý do chính sau.

Thông tin hệ thống cung cấp:
{metadata_context}
"""

        vision_messages = [
            system_prompt,
            (
                f"Yêu cầu người dùng: '{query_text}'. "
                f"Hãy giải thích kết quả thật ngắn gọn, súc tích và chính xác. "
                f"Ưu tiên nêu kết luận mức độ phù hợp trước, sau đó chỉ nêu các ý quan trọng nhất. "
                f"Không lan man, không suy đoán quá mức."
            )
        ]

        image_payloads = []
        for uri in retrieved_uris:
            base64_img = self._encode_image_to_base64(uri)
            if not base64_img:
                continue

            mime_type, _ = mimetypes.guess_type(uri)
            if not mime_type:
                mime_type = "image/jpeg"

            image_payloads.append(f"data:{mime_type};base64,{base64_img}")

        try:
            print("   -> Đang thử gọi API chế độ Vision...")
            vision_prompt = "SYSTEM: " + vision_messages[0] + "\n\nUSER:<image>\n" + vision_messages[1] + "\n\nASSISTANT:"
            response = self._call_fireworks_vision_with_retry(
                model_name=model_name,
                prompt=vision_prompt,
                image_urls=image_payloads,
                max_tokens=800,
                temperature=0.1
            )
            return (response.choices[0].text or "").strip()

        except Exception as e:
            error_msg = str(e)
            short_error = error_msg[:200] if len(error_msg) > 200 else error_msg
            print(f"⚠️ Lỗi Vision API: {short_error}")

            print("🔄 Chuyển sang chế độ giải thích bằng Metadata (Text-Only)...")

            text_fallback_messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Hệ thống đã tìm thấy {len(retrieved_uris)} bức ảnh liên quan đến yêu cầu '{query_text}', "
                        f"nhưng hiện không thể gửi ảnh gốc cho bạn xem. "
                        f"Hãy chỉ dựa trên metadata đã cung cấp để đưa ra nhận xét thật ngắn gọn, chính xác và thận trọng. "
                        f"Nêu kết luận mức độ khớp trước, sau đó chỉ nói các ý quan trọng nhất. "
                        f"Không lan man, không suy đoán quá mức."
                    )
                }
            ]

            try:
                response_text_only = self._call_llm_with_retry(
                    model_name=model_name,
                    messages=text_fallback_messages,
                    max_tokens=600,
                    temperature=0.1
                )
                return (
                    "*(Lưu ý: Server Custom API hiện tại không thể xử lý ảnh trực quan, "
                    "hệ thống tự động chuyển sang giải thích bằng siêu dữ liệu)*\n\n"
                    + self._extract_text_content(response_text_only)
                )
            except Exception as e2:
                short_error2 = str(e2)[:200] if len(str(e2)) > 200 else str(e2)
                return (
                    f"⚠️ Rất tiếc, máy chủ API proxy đang không hoạt động sau nhiều lần thử lại.\n\n"
                    f"**Chi tiết lỗi:** {short_error2}\n\n"
                    f"💡 **Gợi ý:** Hãy kiểm tra lại biến môi trường `OPENAI_BASE_URL` hoặc thử lại sau vài phút."
                )

    # ==========================================
    # UTILS & INDEXING
    # ==========================================
    def convert_images_path_to_absolute(self, images):
        absolute_images = []
        for image in images:
            if not os.path.isabs(image):
                image = os.path.join(os.path.dirname(__file__), image)
            absolute_images.append(image)
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file doesn't exist: {image}")
        return absolute_images

    def index_image(self, source_folder: str = IMAGES_DIR, batch_size: int = 32):
        if not os.path.exists(source_folder):
            print(f"⚠️ Image folder does not exist: {source_folder}")
            return "Folder not found"

        print(f"📂 Scanning images in: {source_folder}")
        ids, uris, metadatas = [], [], []
        existing_ids = set(self.collection.get().get('ids', []))
        count_processed = 0

        for root, _, files in os.walk(source_folder):
            files = sorted(files)
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    file_path = os.path.join(root, file)

                    # SỬA NHẸ:
                    # tránh trùng id nếu cùng tên file ở thư mục khác nhau
                    relative_path = os.path.relpath(file_path, source_folder)
                    item_id = relative_path.replace("\\", "/")

                    if item_id in existing_ids:
                        continue

                    try:
                        date_ts = os.path.getmtime(file_path)
                        date_dt = datetime.fromtimestamp(date_ts)
                        date_int = int(date_dt.strftime("%Y%m%d"))

                        ids.append(item_id)
                        uris.append(file_path)
                        metadatas.append({
                            "date": date_int,
                            "file_name": os.path.basename(file_path),
                            "relative_path": relative_path.replace("\\", "/")
                        })

                        count_processed += 1
                        if count_processed % 100 == 0:
                            print(f"   ... Prepared {count_processed} images")
                    except Exception as e:
                        print(f"⚠️ Error file {file_path}: {e}")

        total_items = len(ids)
        if total_items == 0:
            print("✅ No new images to index.")
            return "No new images"

        print(f"    Start saving {total_items} images to DB (Batch size: {batch_size})...")
        for i in range(0, total_items, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_uris = uris[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            try:
                self.collection.add(ids=batch_ids, uris=batch_uris, metadatas=batch_metadatas)
                print(f"   ✅ Saved batch {i} -> {i + len(batch_ids)}")
            except Exception as e:
                print(f"❌ Error saving batch at index {i}: {e}")

        msg = f"Completed indexing {total_items} images."
        print(msg)
        return msg

    # ==========================================
    # RETRIEVAL EVALUATION
    # ==========================================
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Precision@K = số kết quả đúng trong top-K / K
        """
        retrieved_k = retrieved_ids[:k]
        if k <= 0:
            return 0.0
        hits = sum(1 for item in retrieved_k if item in set(relevant_ids))
        return hits / k

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Recall@K = số kết quả đúng trong top-K / tổng số kết quả đúng thực sự
        """
        relevant_set = set(relevant_ids)
        if len(relevant_set) == 0:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        hits = sum(1 for item in retrieved_k if item in relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def f1_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        F1@K = 2PR / (P + R)
        """
        p = ImageRAG.precision_at_k(retrieved_ids, relevant_ids, k)
        r = ImageRAG.recall_at_k(retrieved_ids, relevant_ids, k)
        if (p + r) == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        RR = 1 / rank của kết quả đúng đầu tiên
        """
        relevant_set = set(relevant_ids)
        for idx, item in enumerate(retrieved_ids, start=1):
            if item in relevant_set:
                return 1.0 / idx
        return 0.0

    @staticmethod
    def mean_reciprocal_rank(list_of_retrieved_ids: List[List[str]], list_of_relevant_ids: List[List[str]]) -> float:
        """
        MRR = trung bình RR trên toàn bộ query
        """
        if not list_of_retrieved_ids:
            return 0.0
        rr_scores = [
            ImageRAG.reciprocal_rank(retrieved, relevant)
            for retrieved, relevant in zip(list_of_retrieved_ids, list_of_relevant_ids)
        ]
        return float(np.mean(rr_scores)) if rr_scores else 0.0

    @staticmethod
    def dcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        DCG@K = sum((rel_i) / log2(i+1))
        Với bài toán relevance nhị phân: rel_i = 1 nếu đúng, ngược lại 0
        """
        relevant_set = set(relevant_ids)
        dcg = 0.0
        for i, item in enumerate(retrieved_ids[:k], start=1):
            rel_i = 1 if item in relevant_set else 0
            dcg += rel_i / math.log2(i + 1)
        return dcg

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        nDCG@K = DCG@K / IDCG@K
        """
        dcg = ImageRAG.dcg_at_k(retrieved_ids, relevant_ids, k)

        ideal_rels = [1] * min(len(relevant_ids), k)
        idcg = 0.0
        for i, rel_i in enumerate(ideal_rels, start=1):
            idcg += rel_i / math.log2(i + 1)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def evaluate_retrieval(
        self,
        queries_with_gt: List[Dict[str, Any]],
        k: int = 5,
        mode: str = "text_to_image"
    ) -> Dict[str, float]:
        """
        queries_with_gt format:
        [
            {
                "query": "golden retriever",
                "relevant_ids": ["dog1.jpg", "dogs/golden_2.jpg"]
            },
            ...
        ]

        mode:
        - "text_to_image"
        - "image_to_image" (query là đường dẫn ảnh)
        """
        precisions = []
        recalls = []
        f1s = []
        ndcgs = []
        all_retrieved = []
        all_relevants = []

        for item in queries_with_gt:
            query = item["query"]
            relevant_ids = item.get("relevant_ids", [])

            if mode == "text_to_image":
                results = self.image_uris(query_text=query, max_results=k)
            elif mode == "image_to_image":
                results = self.image_to_image_retrieval(query_image_path=query, max_results=k)
            else:
                raise ValueError("mode must be either 'text_to_image' or 'image_to_image'")

            retrieved_ids = []
            for r in results:
                metadata = r.get("metadata", {}) or {}
                rid = metadata.get("relative_path") or os.path.basename(r["uri"])
                retrieved_ids.append(rid)

            precisions.append(self.precision_at_k(retrieved_ids, relevant_ids, k))
            recalls.append(self.recall_at_k(retrieved_ids, relevant_ids, k))
            f1s.append(self.f1_at_k(retrieved_ids, relevant_ids, k))
            ndcgs.append(self.ndcg_at_k(retrieved_ids, relevant_ids, k))

            all_retrieved.append(retrieved_ids)
            all_relevants.append(relevant_ids)

        mrr = self.mean_reciprocal_rank(all_retrieved, all_relevants)

        summary = {
            f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
            f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
            f"F1@{k}": float(np.mean(f1s)) if f1s else 0.0,
            f"nDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "MRR": mrr
        }

        return summary

    def calibrate_threshold(
        self,
        queries_with_gt: List[Dict[str, Any]],
        candidate_thresholds: List[float],
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Chỉ dùng khi distance metric = cosine,
        vì khi đó similarity = 1 - distance mới hợp lệ.
        Mục tiêu: tìm threshold similarity tối ưu theo F1.
        """
        if self.distance_metric != "cosine":
            raise ValueError(
                f"Threshold calibration by similarity currently requires cosine metric, got: {self.distance_metric}"
            )

        best_threshold = None
        best_f1 = -1.0
        all_stats = []

        for threshold in candidate_thresholds:
            y_true = []
            y_pred = []

            for item in queries_with_gt:
                query = item["query"]
                relevant_ids = set(item.get("relevant_ids", []))

                results = self.image_uris(query_text=query, max_results=k)

                for r in results:
                    metadata = r.get("metadata", {}) or {}
                    rid = metadata.get("relative_path") or os.path.basename(r["uri"])

                    sim = r.get("similarity")
                    if sim is None:
                        continue

                    pred = 1 if sim >= threshold else 0
                    true = 1 if rid in relevant_ids else 0

                    y_pred.append(pred)
                    y_true.append(true)

            precision, recall, f1 = self._binary_classification_metrics(y_true, y_pred)

            all_stats.append({
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return {
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "all_stats": all_stats
        }

    @staticmethod
    def _binary_classification_metrics(y_true: List[int], y_pred: List[int]):
        """
        Precision / Recall / F1 cho bài toán nhị phân sau thresholding
        """
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1