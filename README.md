# CSE457 - Xử lý Âm thanh và Tiếng nói | Lab 2
## ĐẶC TRƯNG TIẾNG NÓI VÀ NHẬN DẠNG TỪ ĐƠN BẰNG DTW
**Trường Đại học Thủy Lợi - Khoa Công nghệ Thông tin - Bộ môn Trí tuệ Nhân tạo**

Dự án triển khai hoàn chỉnh pipeline nhận dạng tiếng nói từ đơn (Isolated Word Recognition) cho tiếng Việt sử dụng đặc trưng âm học MFCC và thuật toán căn chỉnh thời gian động Dynamic Time Warping (DTW).

---

### 1. Tính năng nổi bật
- **Tiền xử lý âm thanh:** Tự động chuẩn hóa về $16\text{ kHz}$ Mono, 16-bit PCM, chống clipping.
- **Phân tích miền thời gian:** Khung 25ms, bước dịch 10ms, cửa sổ Hamming, tính Short-time Energy, Magnitude, RMS và Zero-Crossing Rate (ZCR).
- **Phát hiện tiếng nói (VAD / Endpoint Detection):** Tự động cắt bỏ silence ở đầu/cuối có chèn vùng đệm an toàn (safety margin 50ms) để không bị xén mất phụ âm yếu.
- **Trích xuất MFCC:** Bộ lọc tiền nhấn Pre-emphasis ($\alpha = 0.97$), FFT 512, ngân hàng 24 bộ lọc Mel, biến đổi logarit, DCT lấy 13 hệ số đầu, và chuẩn hóa Cepstral Mean Normalization (CMN).
- **Tự cài đặt thuật toán DTW:** Cài đặt từ đầu bằng Python/NumPy (không dùng thư viện ngoài) với khoảng cách Euclidean, quy hoạch động ma trận tích lũy $D[i,j]$, truy vết optimal path và chuẩn hóa chi phí theo độ dài đường căn chỉnh $|P|$.
- **Bộ nhận dạng Nearest-Template:** Phân loại từ theo khoảng cách DTW nhỏ nhất, in Top-3 kết quả và hỗ trợ cơ chế Reject Threshold ($\theta$) cho từ lạ.
- **Đánh giá & Thí nghiệm:** Xuất ma trận nhầm lẫn (Confusion Matrix), Accuracy 100%, thực hiện 2 thí nghiệm mở rộng E1 (ảnh hưởng của Endpoint Detection) và E2 (MFCC tĩnh vs MFCC + Delta).

---

### 2. Cấu trúc thư mục
```text
├── dataset/                            # Bộ dữ liệu 25 file chuẩn (5 từ x 5 lần lặp)
│   ├── khong/ (khong_01.wav ... 05.wav)
│   ├── mot/   (mot_01.wav ... 05.wav)
│   ├── hai/   (hai_01.wav ... 05.wav)
│   ├── ba/    (ba_01.wav ... 05.wav)
│   └── bon/   (bon_01.wav ... 05.wav)
├── figures/                            # Biểu đồ kết quả trực quan
│   ├── A_waveform_inspection.png
│   ├── B_energy_zcr_khong.png
│   ├── B_autocorrelation_pitch.png
│   ├── C_endpoint_detection.png
│   ├── D_mfcc_comparison.png
│   ├── E_dtw_same_word.png
│   ├── E_dtw_diff_word.png
│   └── G_confusion_matrix.png
├── user_converted_wav/                 # File wav chuyển đổi từ mp3 của người dùng
├── lab2_main.py                        # Script thực thi toàn bộ pipeline Lab 2
├── recognize.py                        # Script nhận dạng file âm thanh bất kỳ
├── Lab2_Speech_Recognition_DTW.ipynb   # Notebook báo cáo & thực nghiệm
└── results.csv                         # Kết quả đánh giá chi tiết trên tập test
```

---

### 3. Hướng dẫn cài đặt & Chạy chương trình

#### Cài đặt thư viện:
```bash
pip install numpy scipy matplotlib librosa soundfile scikit-learn seaborn pandas edge-tts
```

#### Chạy toàn bộ chương trình và sinh báo cáo:
```bash
python lab2_main.py
```

#### Nhận dạng file âm thanh bất kỳ:
```bash
python recognize.py 1.mp3
# hoặc
python recognize.py dataset/khong/khong_04.wav
```
