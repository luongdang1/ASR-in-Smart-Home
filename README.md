# ASR-in-Smart-Home
SmartHome-ASR is a speech recognition system built on the DeepSpeech2 architecture (originally from Mozilla), trained from scratch with the Fluent Speech Commands Dataset to support smart home applications.

📌 Highlights
🔧 Production-ready: Hệ thống được thiết kế để triển khai dễ dàng trong môi trường thực tế.

🧠 Bi-GRU Architecture: Khai thác kiến trúc RNN hai chiều để nắm bắt ngữ cảnh toàn cục.

🔤 CTC Loss: Huấn luyện không cần căn chỉnh chính xác từng khung âm thanh với ký tự.

📈 Tối ưu hóa cho nhà thông minh: Nhận dạng các lệnh điều khiển như "turn on the lights", "set temperature", v.v.

📂 Dataset
Fluent Speech Commands Dataset

Ngôn ngữ: Tiếng Anh

Dạng dữ liệu: Tập .wav với nhãn dạng câu lệnh

Ví dụ: "turn on the lights in the kitchen"

Tập huấn luyện có sẵn cho nhiều speaker → đa dạng giọng nói

⚙️ Mô hình
🔁 Kiến trúc tổng thể
Feature extraction: STFT → Spectrogram

Encoder: Nhiều tầng Bi-GRU

CTC Loss: Tối ưu hóa học chuỗi không căn chỉnh

<p align="center"> <img src="Picture_1751183842" alt="Bi-GRU Architecture" width="600"/> </p>
✨ CTC Loss – Connectionist Temporal Classification
Cho phép huấn luyện không cần gán nhãn từng frame âm thanh với ký tự cụ thể.

Quá trình suy luận:

Dự đoán chuỗi ký tự có cả blank và lặp lại.

Merge duplicates: loại bỏ ký tự lặp liên tiếp.

Remove blanks: loại bỏ ký hiệu blank để tạo ra chuỗi cuối cùng.

<p align="center"> <img src="Picture_1167238156" alt="CTC Explanation" width="600"/> </p>
📈 Kết quả căn chỉnh (Alignment Example)
Hệ thống có thể căn chỉnh từ với vùng âm thanh tương ứng.

Hình minh họa vùng phát âm tương ứng cho từng từ (highlight nền màu).

<p align="center"> <img src="Picture_226716643" alt="Waveform Alignment" width="600"/> </p>
🚀 Cài đặt
Yêu cầu:
Python ≥ 3.10

Conda (đề xuất dùng Miniconda)

CUDA 12.1

PyTorch ≥ 2.2.2 + cu121

Libsox (âm thanh)

Các bước:
bash
Sao chép
Chỉnh sửa
# Clone dự án
git clone https://github.com/your-username/SmartHome-ASR.git
cd SmartHome-ASR

# Tạo môi trường Conda
conda create -n smarthome-asr python=3.10
conda activate smarthome-asr

# Cài đặt phụ thuộc
pip install -r requirements.txt

# Cài đặt CUDA, Torch (nếu dùng GPU)
pip install torch==2.2.2+cu121 torchaudio==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Cài đặt sox nếu gặp lỗi liên quan
conda install conda-forge::sox
🧪 Huấn luyện & Inference
python
Sao chép
Chỉnh sửa
from deepspeech_model import DeepSpeech2Model

model = DeepSpeech2Model.load_pretrained('models/deepspeech2_fsc.pth')
result = model.transcribe('audio/turn_on_light.wav')
print(result['text'])  # Output: turn on the light
🗂 Cấu trúc thư mục
bash
Sao chép
Chỉnh sửa
SmartHome-ASR/
├── data/                   # Dữ liệu đầu vào (.wav, labels)
├── models/                 # Trọng số mô hình đã huấn luyện
├── notebooks/              # Phân tích dữ liệu & inference
├── scripts/                # Script huấn luyện, test
├── deepspeech_model.py     # Cấu trúc DeepSpeech2
├── requirements.txt
└── README.md
📚 Tài liệu tham khảo
Mozilla DeepSpeech

Fluent Speech Commands Dataset

CTC Loss paper

📌 Giấy phép
Dự án này được phát hành dưới MIT License.
