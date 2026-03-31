import os
import sys
import tempfile
import streamlit as st

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add project root to path for importsnsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))n

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from src.engine import ImageRAG
from downstream_inference.settings import IMAGES_DIR, OPENAI_TEXT_MODEL_ID, OPENAI_VISION_MODEL_ID

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & KHỞI TẠO MODEL
# ==========================================
st.set_page_config(page_title="Hệ thống Truy vấn Ảnh", layout="wide")
st.title("🔍 Hệ thống Truy vấn Ảnh đa phương thức (CBIR + RAG)")

st.markdown("""
<style>
.rank-badge {
    background-color: #5cb85c;
    color: white;
    padding: 6px 14px;
    border-radius: 4px;
    font-weight: bold;
    display: inline-block;
    margin-bottom: 10px;
}
.img-metrics {
    text-align: center;
    font-family: sans-serif;
    font-size: 14px;
    line-height: 1.6;
    margin-top: 10px;
    color: #e0e0e0;
}
.intent-box {
    background: #1e1e1e;
    padding: 12px 14px;
    border-radius: 8px;
    margin-bottom: 12px;
    border: 1px solid #333;
}
.ai-summary-box {
    background: #0f172a;
    padding: 14px 16px;
    border-radius: 10px;
    border: 1px solid #334155;
    margin-top: 8px;
    line-height: 1.7;
}
.small-note {
    color: #94a3b8;
    font-size: 13px;
    margin-top: -6px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return ImageRAG()

with st.spinner("Đang tải mô hình CLIP và kết nối Database..."):
    engine = load_engine()

top_left, _ = st.columns([1, 5])
with top_left:
    if st.button("Ping provider"):
        with st.spinner("Đang kiểm tra provider..."):
            ping_result = engine.ping_provider()
        status_code = ping_result.get("status_code")
        status_label = f"HTTP {status_code}" if status_code is not None else "No HTTP status"
        if ping_result.get("ok"):
            st.success(f"{status_label}: {ping_result.get('message')}")
        else:
            st.error(f"{status_label}: {ping_result.get('message')}")

# ==========================================
# 2. HÀM HỖ TRỢ
# ==========================================
def format_date_from_metadata(metadata):
    if not metadata:
        return "Không rõ"
    date_value = metadata.get("date")
    if date_value is None:
        return "Không rõ"
    date_str = str(date_value)
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str

def format_score_lines(data):
    """
    Hiển thị score theo đúng metric:
    - Nếu engine trả similarity hợp lệ thì hiển thị similarity
    - Luôn hiển thị distance nếu có
    - Hiển thị metric để tránh hiểu nhầm
    """
    metric = data.get("metric", "unknown")

    similarity_line = ""
    if data.get("similarity") is not None:
        similarity_line = f"Similarity ({metric}): {data['similarity']:.4f}<br>"

    distance_line = ""
    if data.get("distance") is not None:
        distance_line = f"Distance ({metric}): {data['distance']:.4f}<br>"

    metric_line = f"Metric: {metric}<br>"

    return similarity_line, distance_line, metric_line

def render_result_grid(found_data, columns_per_row=2):
    cols = st.columns(columns_per_row)

    for i, data in enumerate(found_data):
        with cols[i % columns_per_row]:
            rank = data.get("rank", i + 1)
            st.markdown(f'<div class="rank-badge">Rank #{rank}</div>', unsafe_allow_html=True)

            if data.get("uri") and os.path.exists(data["uri"]):
                st.image(data["uri"], use_container_width=True)
            else:
                st.warning("Không thể hiển thị ảnh do đường dẫn không hợp lệ.")
                continue

            metadata = data.get("metadata") or {}
            file_name = metadata.get("file_name") or os.path.basename(data["uri"])
            date_str = format_date_from_metadata(metadata)

            similarity_line, distance_line, metric_line = format_score_lines(data)

            st.markdown(f"""
            <div class="img-metrics">
                {similarity_line}
                {distance_line}
                {metric_line}
                Date: {date_str}<br>
                File: {file_name}
            </div><br>
            """, unsafe_allow_html=True)

def render_ai_explanation(explanation: str):
    """
    Hiển thị phần giải thích AI theo cách gọn, dễ đọc hơn.
    """
    if not explanation:
        st.warning("Chưa có nội dung giải thích từ AI.")
        return

    safe_text = explanation.strip().replace("\n", "<br>")
    st.markdown(f"""
    <div class="ai-summary-box">
        {safe_text}
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR: QUẢN LÝ DỮ LIỆU
# ==========================================
with st.sidebar:
    st.header("⚙️ Quản lý Knowledge Base")
    st.write(f"Thư mục ảnh: `{IMAGES_DIR}`")

    if st.button("🚀 Chạy Index Dữ Liệu (Ingestion)"):
        with st.spinner("Đang quét và nhúng vector vào ChromaDB..."):
            try:
                msg = engine.index_image(batch_size=32)
                st.success(msg)
            except Exception as e:
                st.error(f"Lỗi khi indexing dữ liệu: {e}")

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
tab1, tab2 = st.tabs([
    "📝 Tìm bằng Text & Giải thích (Text-to-Image)",
    "🖼️ Tìm bằng Ảnh (Image-to-Image)"
])

# ----- TAB 1: TEXT-TO-IMAGE -----
with tab1:
    st.markdown("### Nhập mô tả để tìm ảnh và nhận phần tóm tắt ngắn từ AI")
    query_text = st.text_input(
        "Ví dụ: Find me some black dogs which is taken this year",
        key="text_query"
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        top_k = st.number_input("Số lượng kết quả", min_value=1, max_value=20, value=4, step=1)
    with col_b:
        use_ai_intent = st.checkbox("Dùng AI để bóc tách ý định và dịch sang tiếng Anh", value=True)

    if st.button("Tìm kiếm & Phân tích", type="primary"):
        if not query_text.strip():
            st.error("Vui lòng nhập câu truy vấn!")
        else:
            intent = {
                "english_query": query_text,
                "start_date": None,
                "end_date": None,
                "max_results": int(top_k)
            }

            if use_ai_intent:
                with st.spinner("Đang phân tích yêu cầu..."):
                    try:
                        intent = engine.extract_query_intent(query_text, model_name=OPENAI_TEXT_MODEL_ID)
                    except Exception as e:
                        intent = {
                            "english_query": query_text,
                            "start_date": None,
                            "end_date": None,
                            "max_results": int(top_k),
                            "_llm_error": str(e)
                        }

                if "_llm_error" in intent:
                    st.warning(
                        "⚠️ Provider có phản hồi nhưng không trả về JSON hợp lệ để phân tích. "
                        "Hệ thống sẽ tìm trực tiếp bằng từ khóa gốc.\n\n"
                        f"*Chi tiết: {intent['_llm_error']}*"
                    )
                    raw_output = intent.get("_llm_raw_output")
                    if raw_output:
                        with st.expander("Xem phản hồi thô từ model"):
                            st.code(raw_output)

                if not intent.get("max_results"):
                    intent["max_results"] = int(top_k)
            else:
                intent["max_results"] = int(top_k)

            st.markdown("#### 🔎 Ý định truy vấn đã được phân tích")
            st.markdown(f"""
            <div class="intent-box">
                <b>English query:</b> {intent.get("english_query", query_text)}<br>
                <b>Start date:</b> {intent.get("start_date")}<br>
                <b>End date:</b> {intent.get("end_date")}<br>
                <b>Max results:</b> {intent.get("max_results", top_k)}
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Đang truy xuất ảnh từ ChromaDB..."):
                try:
                    found_data = engine.retrieve_images(
                        user_query=intent.get("english_query", query_text),
                        start_date=intent.get("start_date"),
                        end_date=intent.get("end_date"),
                        max_results=int(intent.get("max_results", top_k))
                    )
                except Exception as e:
                    found_data = []
                    st.error(f"Lỗi trong quá trình truy xuất ảnh: {e}")

            if not found_data:
                st.warning("Không tìm thấy ảnh nào khớp với mô tả trong cơ sở dữ liệu.")
            else:
                metric = found_data[0].get("metric", "unknown")
                st.success(f"Đã tìm thấy {len(found_data)} ảnh tương đồng!")

                if metric != "cosine":
                    st.info(
                        f"Hệ thống đang dùng metric '{metric}'. "
                        "Similarity chỉ được diễn giải trực tiếp khi metric là cosine; "
                        "vì vậy hãy ưu tiên đọc distance và thứ hạng (rank)."
                    )

                render_result_grid(found_data, columns_per_row=2)

                st.markdown("---")
                st.markdown("### 🤖 Phần tóm tắt kết quả")
                st.markdown(
                    '<div class="small-note">Phần dưới đây là tóm tắt, tập trung vào mức độ phù hợp và các ý quan trọng nhất.</div>',
                    unsafe_allow_html=True
                )

                with st.spinner("Đang tạo phần tóm tắt ..."):
                    try:
                        explanation = engine.rag_generate_explanation(
                            query_text=query_text,
                            retrieved_data=found_data,
                            model_name=OPENAI_VISION_MODEL_ID
                        )
                        render_ai_explanation(explanation)
                    except Exception as e:
                        st.error(f"Lỗi khi tạo phần giải thích: {e}")

# ----- TAB 2: IMAGE-TO-IMAGE -----
with tab2:
    st.markdown("### Tải lên một bức ảnh để tìm các hình ảnh tương tự")
    uploaded_file = st.file_uploader(
        "Chọn ảnh từ máy của bạn...",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload"
    )

    top_k_img = st.number_input(
        "Số lượng ảnh tương tự cần lấy",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        key="img_top_k"
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Ảnh bạn vừa tải lên", width=300)

        if st.button("Tìm ảnh tương tự", key="img_search_btn"):
            temp_path = None
            try:
                with st.spinner("Đang tính toán vector hình ảnh..."):
                    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_path = tmp_file.name

                    found_data = engine.image_to_image_retrieval(
                        query_image_path=temp_path,
                        max_results=int(top_k_img)
                    )

                if not found_data:
                    st.warning("Không tìm thấy ảnh tương đồng.")
                else:
                    metric = found_data[0].get("metric", "unknown")
                    st.success(f"Đã tìm thấy {len(found_data)} ảnh tương tự!")

                    if metric != "cosine":
                        st.info(
                            f"Hệ thống đang dùng metric '{metric}'. "
                            "Similarity không nên suy diễn trực tiếp nếu metric không phải cosine."
                        )

                    render_result_grid(found_data, columns_per_row=2)

            except Exception as e:
                st.error(f"Lỗi khi truy vấn ảnh-đến-ảnh: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)