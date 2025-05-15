import os
import re
import json
import faiss
import numpy as np
import urllib.parse
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load model chỉ 1 lần
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index đã build sẵn
index = faiss.read_index("tuyensinh.index")

# Load nội dung các đoạn từ chunks.json
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Cấu hình Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # KHÔNG dùng key mặc định
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

# Lưu lịch sử hội thoại
conversation_history = []

def search_context(query, top_k=5):
    """
    Tìm các đoạn văn bản liên quan nhất từ FAISS
    """
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def answer_question(query):
    """
    Tạo prompt và gọi Gemini để sinh phản hồi
    """
    contexts = search_context(query)
    
    # Lấy lịch sử hội thoại gần nhất
    history_text = ""
    for q, a in conversation_history[-3:]:
        history_text += f"Câu hỏi: {q}\nTrả lời: {a}\n\n"

    # Prompt chính
    prompt = f"""
    Bạn là chuyên gia tư vấn tuyển sinh ĐH Đồng Tháp. Chỉ trả lời các câu hỏi liên quan đến tuyển sinh ĐH Đồng Tháp dựa trên thông tin được cung cấp. 
    Bạn không được phép đưa ra thông tin mà không có trong ngữ cảnh.
    
    **Hướng dẫn:**
    - Trả lời trực tiếp, không vòng vo.
    - Trình bày rõ ràng từng đoạn.
    - Tô đậm các từ khóa chính bằng **.
    
    **Lịch sử hội thoại (nếu có):**
    {history_text}

    **Ngữ cảnh tuyển sinh:**
    {chr(10).join(contexts)}

    **Câu hỏi:** {query}

    **Trả lời:**
    """

    # Gọi Gemini
    response = gemini.generate_content(prompt)
    answer = response.text

    # Chuyển link thành thẻ <a>
    def clean_url(match):
        url = match.group(0)
        return f'<a href="{url}" target="_blank" class="text-blue-600 underline hover:text-blue-800">{url}</a>'

    answer = re.sub(r'(https?://[^\s<>"\'&]+)', clean_url, answer)
    answer = re.sub(r'\*\*([^\*]+?)\*\*', r'<strong>\1</strong>', answer)

    # Chuyển thành HTML đoạn văn
    paragraphs = [p.strip() for p in answer.split('\n') if p.strip()]
    formatted_answer = '<br>'.join([f'<p>{p}</p>' for p in paragraphs])

    # Lưu lịch sử
    conversation_history.append((query, response.text))
    if len(conversation_history) > 10:
        conversation_history.pop(0)

    return formatted_answer
