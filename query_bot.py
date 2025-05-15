import json
import re
import urllib.parse
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np

# Load model và dữ liệu
model = SentenceTransformer("all-MiniLM-L6-v2")

# Đọc dữ liệu từ file tuyensinh.txt
with open("tuyensinh.txt", "r", encoding="utf-8") as f:
    raw_data = f.read()

# Chia dữ liệu thành các đoạn nhỏ
def split_into_chunks(text, max_length=500):
    # Tách theo các mục chính (số + dấu chấm)
    sections = re.split(r'\n\d+\.', text)
    chunks = []
    current_chunk = ""
    
    for section in sections:
        if len(current_chunk) + len(section) <= max_length:
            current_chunk += section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

chunks = split_into_chunks(raw_data)

# Cấu hình Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyALa6vneqEcI339z5m2shZKx8k15wQ2iyA')
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

# Lưu lịch sử hội thoại
conversation_history = []

def search_context(query, top_k=7):
    """
    Tìm các đoạn văn bản liên quan đến câu hỏi dựa trên độ tương đồng ngữ nghĩa
    """
    # Mã hóa câu hỏi
    query_embedding = model.encode([query])[0]
    
    # Mã hóa tất cả các đoạn văn bản
    chunk_embeddings = model.encode(chunks)
    
    # Tính độ tương đồng cosine
    similarities = []
    for chunk_embedding in chunk_embeddings:
        similarity = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
        similarities.append(similarity)
    
    # Lấy top_k đoạn văn bản có độ tương đồng cao nhất
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def answer_question(query):
    """
    Tạo prompt với lịch sử hội thoại và gửi yêu cầu đến Gemini
    """
    contexts = search_context(query)
    # Giới hạn lịch sử hội thoại (chỉ lấy 3 cặp gần nhất)
    history_text = ""
    for q, a in conversation_history[-3:]:
        history_text += f"Câu hỏi: {q}\nTrả lời: {a}\n\n"

    prompt = f"""
    Bạn là chuyên gia tư vấn tuyển sinh ĐH Đồng Tháp. Chỉ trả lời các câu hỏi liên quan đến tuyển sinh ĐH Đồng Tháp dựa trên thông tin được cung cấp. 
    Bạn có thể trò chuyện phiếm với người hỏi nhưng Không cung cấp thông tin ngoài phạm vi này. Bạn không được phép tự ý đưa ra thông tin mà không có trong ngữ cảnh.
    Trả lời như tôi chưa cung cấp thông tin gì cho bạn.

    **Hướng dẫn:**
    - Trả lời trực tiếp, và chính xác vào câu hỏi, không thêm câu mở đầu hoặc diễn giải không cần thiết như 'tôi xin cung cấp', 'dựa trên dữ liệu', hoặc tương tự.
    - Đảm bảo câu trả lời đầy đủ, bao quát tất cả khía cạnh liên quan đến câu hỏi, sử dụng toàn bộ thông tin từ ngữ cảnh.
    - Nếu câu hỏi chung chung (như "Thông tin tuyển sinh"), cung cấp tổng quan nêu rõ về các mục chính: thời gian nhận hồ sơ, phương thức xét tuyển, học phí, chính sách học bổng, danh sách ngành tiêu biểu, và các thông báo quan trọng.
    - Phân đoạn câu trả lời thành các đoạn văn ngắn gọn, mỗi đoạn tập trung vào một ý chính.
    - Sử dụng ký hiệu ** để in đậm các thông tin quan trọng như ngày tháng, số liệu, hoặc từ khóa chính.
    - Nếu câu hỏi liên quan đến lịch sử hội thoại, tham khảo ngữ cảnh từ lịch sử nhưng ưu tiên thông tin từ dữ liệu tuyển sinh.

    **Lịch sử hội thoại (nếu liên quan):**
    {history_text}

    **Thông tin tuyển sinh ĐH Đồng Tháp:**
    {chr(10).join(contexts)}

    **Câu hỏi hiện tại:** {query}

    **Trả lời:**
    """

    response = gemini.generate_content(prompt)
    answer = response.text

    # Hàm làm sạch URL
    def clean_url(match):
        url = match.group(0)
        decoded_url = urllib.parse.unquote(url)  # Giải mã URL
        # Loại bỏ các thẻ HTML không hợp lệ trong URL
        cleaned_url = re.sub(r'</?\w+>', '', decoded_url)
        return f'<a href="{cleaned_url}" target="_blank" class="text-blue-600 underline hover:text-blue-800">{cleaned_url}</a>'

    # Chuyển đổi URL thành thẻ <a> trước
    url_pattern = r'(https?://[^\s<>"\'&]+)'
    answer = re.sub(url_pattern, clean_url, answer)

    # Chuyển đổi định dạng in đậm sau
    answer = re.sub(r'\*\*([^\*]+?)\*\*', r'<strong>\1</strong>', answer)

    # Phân tách câu trả lời thành các đoạn văn
    paragraphs = [p.strip() for p in answer.split('\n') if p.strip()]
    formatted_answer = '<br>'.join([f'<p>{p}</p>' for p in paragraphs])

    # Lưu câu hỏi và câu trả lời (chưa định dạng) vào lịch sử
    conversation_history.append((query, response.text))  # Lưu câu trả lời gốc từ Gemini
    
    # Giới hạn lịch sử hội thoại (chỉ giữ 10 cặp gần nhất)
    if len(conversation_history) > 10:
        conversation_history.pop(0)

    return formatted_answer