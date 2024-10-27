from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

app = Flask(__name__)

# Bật CORS cho ứng dụng
CORS(app)  # Thêm dòng này để bật CORS cho tất cả các nguồn

# Khởi tạo tokenizer và mô hình BART
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Kiểm tra xem có GPU không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_summary(input_text, model, tokenizer, device):
    # Tokenize văn bản đầu vào
    inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)
    # Tạo tóm tắt
    summary_ids = model.generate(inputs.input_ids.to(device), max_length=150, num_beams=4, early_stopping=True)
    # Giải mã tóm tắt
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    # Lấy dữ liệu JSON từ yêu cầu
    data = request.get_json()
    input_text = data.get('text', '')

    # Kiểm tra đầu vào
    if not input_text:
        return jsonify({"error": "Input text is required."}), 400

    try:
        # Gọi hàm tóm tắt
        summary = generate_summary(input_text, model, tokenizer, device)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
