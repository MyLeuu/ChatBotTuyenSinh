import os
from flask import Flask, request, render_template, Response
from query_bot import answer_question

# Khởi tạo Flask app
app = Flask(__name__)

# Route cho trang chủ
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form.get("query")
        if query.strip().lower() in ["exit", "quit"]:
            return Response("👋 Tạm biệt!", mimetype="text/plain")
        answer = answer_question(query)
        return Response(answer, mimetype="text/html")
    return render_template("index.html")

if __name__ == "__main__":
    print("🤖 Chatbot tư vấn tuyển sinh ĐH Đồng Tháp đang chạy...")
    app.run(debug=True) 