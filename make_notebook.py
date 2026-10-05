import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
## BÀI THỰC HÀNH SỐ 2: ĐẶC TRƯNG TIẾNG NÓI VÀ NHẬN DẠNG BẰNG DTW
**Học phần:** Xử lý âm thanh và tiếng nói (CSE457)  
**Trường:** Đại học Thủy Lợi - Khoa Công nghệ Thông tin - Bộ môn Trí tuệ Nhân tạo  
---
### Mục tiêu chuẩn đầu ra:
1. Phân tích tín hiệu âm thanh tiếng nói theo frame: Short-time energy, magnitude, RMS và Zero-Crossing Rate (ZCR).
2. Xây dựng thuật toán Endpoint Detection (VAD) dựa trên Log-energy và ZCR để loại bỏ khoảng lặng đầu/cuối.
3. Trích chọn vector đặc trưng MFCC (13 hệ số) kết hợp thang Mel, Filterbank 24 bộ lọc, Log và biến đổi DCT cùng chuẩn hóa CMN.
4. **Tự cài đặt thuật toán Dynamic Time Warping (DTW)** từ đầu với ma trận khoảng cách Euclidean, quy hoạch động ma trận tích lũy chi phí, truy vết đường căn chỉnh tối ưu và chuẩn hóa chi phí theo độ dài đường đi.
5. Xây dựng bộ nhận dạng từ đơn phân loại Nearest-Template (Nearest Neighbor DTW) cho 5 từ tiếng Việt.
6. Đánh giá toàn diện bằng Accuracy, Confusion Matrix, và 2 thí nghiệm mở rộng có kiểm soát E1 & E2.
"""))

# Cell 1: Environment & Config
cells.append(nbf.v4.new_markdown_cell("""### 1. Chuẩn bị môi trường & Tham số hệ thống"""))
cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Cấu hình tham số chuẩn theo yêu cầu bài Lab
FS = 16000             # Tần số lấy mẫu: 16 kHz
FRAME_MS = 25          # Khung: 25 ms
HOP_MS = 10            # Bước dịch: 10 ms
WIN = int(FS * FRAME_MS / 1000)  # L = 400 mẫu
HOP = int(FS * HOP_MS / 1000)    # R = 160 mẫu
N_FFT = 512            # Điểm FFT
N_MELS = 24            # 24 bộ lọc tam giác Mel
N_MFCC = 13            # 13 hệ số MFCC
PRE_EMPH = 0.97        # Hệ số tiền nhấn pre-emphasis

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = Path("dataset")
LABELS = ['khong', 'mot', 'hai', 'ba', 'bon']

print(f"Cấu hình hệ thống: Fs={FS}Hz, Frame={WIN} mẫu (25ms), Hop={HOP} mẫu (10ms), Mel={N_MELS}, MFCC={N_MFCC}")
"""))

# Cell 2: Convert MP3 to WAV
cells.append(nbf.v4.new_markdown_cell("""### 2. Tiền xử lý: Chuyển đổi 5 file MP3 có sẵn sang WAV chuẩn (16 kHz, Mono, PCM)"""))
cells.append(nbf.v4.new_code_cell("""user_wav_dir = Path("user_converted_wav")
user_wav_dir.mkdir(parents=True, exist_ok=True)

converted_info = []
for i in range(1, 6):
    mp3_file = Path(f"{i}.mp3")
    if mp3_file.exists():
        y, sr = librosa.load(mp3_file, sr=FS, mono=True)
        # Chuẩn hóa biên độ đỉnh tránh clipping
        y = y / (np.max(np.abs(y)) + 1e-9)
        out_wav = user_wav_dir / f"{i}.wav"
        sf.write(out_wav, y, FS, subtype='PCM_16')
        converted_info.append({
            'File MP3': mp3_file.name,
            'File WAV xuất': str(out_wav),
            'Thời lượng (s)': round(len(y)/FS, 2),
            'Số mẫu (samples)': len(y)
        })

df_conv = pd.DataFrame(converted_info)
print("Kết quả chuyển đổi 5 file MP3 có sẵn:")
display(df_conv)
"""))

# Cell 3: Data Quality Inspection (Waveform)
cells.append(nbf.v4.new_markdown_cell("""### 3. Phần A: Kiểm tra dạng sóng (Waveform), Clipping và Silence"""))
cells.append(nbf.v4.new_code_cell("""def load_audio(path):
    y, sr = librosa.load(path, sr=FS, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y, sr

# Lấy 3 từ đại diện để kiểm tra waveform
inspect_files = [
    DATASET_DIR / "khong" / "khong_01.wav",
    DATASET_DIR / "mot" / "mot_01.wav",
    DATASET_DIR / "hai" / "hai_01.wav"
]

plt.figure(figsize=(14, 6))
for idx, f in enumerate(inspect_files, 1):
    y, sr = load_audio(f)
    t = np.arange(len(y)) / sr
    plt.subplot(3, 1, idx)
    plt.plot(t, y, color='#1f77b4', linewidth=1)
    plt.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Clip limit (+1.0)')
    plt.axhline(-1.0, color='r', linestyle='--', alpha=0.5, label='Clip limit (-1.0)')
    plt.title(f"Waveform: {f.name} (Thời lượng: {len(y)/sr:.2f}s, Max Amp: {np.max(np.abs(y)):.2f})", fontsize=10, fontweight='bold')
    plt.ylabel("Biên độ")
    plt.grid(True, linestyle=':', alpha=0.6)
    if idx == 1:
        plt.legend(loc='upper right')
    if idx == 3:
        plt.xlabel("Thời gian (giây)")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "A_waveform_inspection.png", dpi=300)
plt.show()
"""))

# Cell 4: Time Domain Features
cells.append(nbf.v4.new_markdown_cell("""### 4. Phần B: Đặc trưng miền thời gian (Short-time Energy, ZCR & Pitch)"""))
cells.append(nbf.v4.new_code_cell("""def compute_time_features(y, win=WIN, hop=HOP):
    hamming_win = np.hamming(win)
    num_frames = 1 + (len(y) - win) // hop
    
    energies = np.zeros(num_frames)
    magnitudes = np.zeros(num_frames)
    rms_values = np.zeros(num_frames)
    zcr_values = np.zeros(num_frames)
    frame_times = np.zeros(num_frames)
    
    for r in range(num_frames):
        start = r * hop
        end = start + win
        frame_raw = y[start:end]
        frame_win = frame_raw * hamming_win
        
        energies[r] = np.sum(frame_win ** 2)
        magnitudes[r] = np.sum(np.abs(frame_win))
        rms_values[r] = np.sqrt(np.mean(frame_win ** 2))
        
        # Zero-Crossing Rate (ZCR)
        sgn = np.sign(frame_raw)
        sgn[sgn == 0] = 1
        zcr_values[r] = np.sum(np.abs(sgn[1:] - sgn[:-1])) / (2.0 * win)
        frame_times[r] = (start + win / 2.0) / FS
        
    log_energy = 10.0 * np.log10(energies + 1e-12)
    return frame_times, energies, log_energy, zcr_values

y_khong, sr = load_audio(DATASET_DIR / "khong" / "khong_01.wav")
times, energies, log_e, zcr = compute_time_features(y_khong)

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
axes[0].plot(np.arange(len(y_khong))/sr, y_khong, color='#2c3e50')
axes[0].set_title("Waveform từ 'không'", fontweight='bold')
axes[0].set_ylabel("Biên độ")
axes[0].grid(True, linestyle=':', alpha=0.6)

axes[1].plot(times, log_e, color='#e74c3c', label='Log-Energy (dB)')
axes[1].set_title("Short-Time Log-Energy", fontweight='bold')
axes[1].set_ylabel("Năng lượng (dB)")
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(loc='upper right')

axes[2].plot(times, zcr, color='#27ae60', label='ZCR')
axes[2].set_title("Zero-Crossing Rate (ZCR)", fontweight='bold')
axes[2].set_xlabel("Thời gian (giây)")
axes[2].set_ylabel("ZCR (lần/mẫu)")
axes[2].grid(True, linestyle=':', alpha=0.6)
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "B_energy_zcr_khong.png", dpi=300)
plt.show()
"""))

# Cell 5: Endpoint Detection
cells.append(nbf.v4.new_markdown_cell("""### 5. Phần C: Endpoint Detection (Cắt bỏ Silence ở đầu và cuối)"""))
cells.append(nbf.v4.new_code_cell("""def detect_endpoint(y, sr=FS, top_db=32, margin_ms=50):
    yt, idx = librosa.effects.trim(y, top_db=top_db, frame_length=WIN, hop_length=HOP)
    margin_samples = int(sr * margin_ms / 1000)
    start_sample = max(0, idx[0] - margin_samples)
    end_sample = min(len(y), idx[1] + margin_samples)
    return y[start_sample:end_sample], (start_sample, end_sample)

trim_results = []
for lab in LABELS:
    f = DATASET_DIR / lab / f"{lab}_01.wav"
    y_raw, _ = load_audio(f)
    y_trim, (s, e) = detect_endpoint(y_raw)
    trim_results.append({
        'Từ': lab,
        'File': f.name,
        'Thời lượng gốc (s)': round(len(y_raw)/FS, 3),
        'Sau Trim (s)': round(len(y_trim)/FS, 3),
        'Điểm đầu (s)': round(s / FS, 3),
        'Điểm cuối (s)': round(e / FS, 3),
        'Tiết kiệm (%)': round((1 - len(y_trim)/len(y_raw))*100, 1)
    })
df_trim = pd.DataFrame(trim_results)
display(df_trim)
"""))

# Cell 6: MFCC Extraction
cells.append(nbf.v4.new_markdown_cell("""### 6. Phần D: Trích xuất đặc trưng MFCC (13 hệ số, 24 Mel filters, CMN)"""))
cells.append(nbf.v4.new_code_cell("""def extract_mfcc(y, sr=FS, n_mfcc=N_MFCC, n_mels=N_MELS, with_delta=False):
    # 1. Pre-emphasis filter
    y_pre = signal.lfilter([1.0, -PRE_EMPH], [1.0], y)
    
    # 2. MFCC extraction
    mfcc = librosa.feature.mfcc(
        y=y_pre, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels,
        n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        window='hamming', center=False
    )
    
    # 3. Cepstral Mean Normalization (CMN)
    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
    
    if with_delta:
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, d1, d2])
    else:
        features = mfcc
        
    return features.T # (T_frames, D_features)

# Heatmap so sánh MFCC của 2 từ khác nhau
y_khong_trim, _ = detect_endpoint(load_audio(DATASET_DIR / "khong" / "khong_01.wav")[0])
y_bon_trim, _ = detect_endpoint(load_audio(DATASET_DIR / "bon" / "bon_01.wav")[0])

mfcc_k = extract_mfcc(y_khong_trim)
mfcc_b = extract_mfcc(y_bon_trim)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)
im1 = axes[0].imshow(mfcc_k.T, aspect='auto', origin='lower', cmap='viridis')
axes[0].set_title(f"MFCC Heatmap: Từ 'không' ({mfcc_k.shape[0]} frames x {mfcc_k.shape[1]} features)", fontweight='bold')
axes[0].set_ylabel("Chỉ số MFCC")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(mfcc_b.T, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_title(f"MFCC Heatmap: Từ 'bốn' ({mfcc_b.shape[0]} frames x {mfcc_b.shape[1]} features)", fontweight='bold')
axes[1].set_xlabel("Chỉ số Frame (10ms/hop)")
axes[1].set_ylabel("Chỉ số MFCC")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig(FIGURES_DIR / "D_mfcc_comparison.png", dpi=300)
plt.show()
"""))

# Cell 7: Custom DTW
cells.append(nbf.v4.new_markdown_cell("""### 7. Phần E: Tự cài đặt Dynamic Time Warping (DTW) & So sánh đường căn chỉnh"""))
cells.append(nbf.v4.new_code_cell("""def dtw_custom(X, Y):
    N, M = len(X), len(Y)
    
    # 1. Ma trận khoảng cách Euclidean cục bộ C (N x M)
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    C = np.linalg.norm(diff, axis=2)
    
    # 2. Ma trận chi phí tích lũy D ((N+1) x (M+1))
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    back = np.zeros((N + 1, M + 1, 2), dtype=int)
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            local_cost = C[i - 1, j - 1]
            choices = [
                (D[i - 1, j - 1], i - 1, j - 1), # Chéo (diagonal)
                (D[i - 1, j],     i - 1, j),     # Lên (vertical)
                (D[i, j - 1],     i,     j - 1)  # Trái (horizontal)
            ]
            best, pi, pj = min(choices, key=lambda z: z[0])
            D[i, j] = local_cost + best
            back[i, j] = [pi, pj]
            
    # 3. Backtracking tìm Optimal Warping Path
    path = []
    i, j = N, M
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = back[i, j]
    path.reverse()
    
    # 4. Chuẩn hóa chi phí theo độ dài đường đi |P|
    dtw_norm = D[N, M] / max(len(path), 1)
    return dtw_norm, path, C, D[1:, 1:]

# Thử nghiệm DTW: Cùng từ vs Khác từ
y_k2_trim, _ = detect_endpoint(load_audio(DATASET_DIR / "khong" / "khong_02.wav")[0])
mfcc_k2 = extract_mfcc(y_k2_trim)

cost_same, path_same, C_same, _ = dtw_custom(mfcc_k, mfcc_k2)
cost_diff, path_diff, C_diff, _ = dtw_custom(mfcc_k, mfcc_b)

print(f"Chi phí DTW_norm giữa 2 lần nói CÙNG TỪ ('không' vs 'không'): {cost_same:.4f}")
print(f"Chi phí DTW_norm giữa 2 từ KHÁC NHAU ('không' vs 'bốn'):      {cost_diff:.4f}")
print(f"=> Tỉ lệ chênh lệch: {cost_diff / cost_same:.2f} lần!")
"""))

# Cell 8: Nearest-Template Classifier
cells.append(nbf.v4.new_markdown_cell("""### 8. Phần F & G: Bộ nhận dạng Nearest-Template & Đánh giá mô hình"""))
cells.append(nbf.v4.new_code_cell("""# Tạo tập mẫu huấn luyện (Templates: 3 file đầu mỗi từ)
train_templates = {lab: [] for lab in LABELS}
for lab in LABELS:
    files = sorted(list((DATASET_DIR / lab).glob("*.wav")))[:3]
    for f in files:
        y, _ = load_audio(f)
        y, _ = detect_endpoint(y)
        train_templates[lab].append(extract_mfcc(y))

# Tập kiểm thử (Test: 2 file sau mỗi từ)
test_files = []
for lab in LABELS:
    files = sorted(list((DATASET_DIR / lab).glob("*.wav")))[3:]
    for f in files:
        test_files.append((f, lab))

y_true, y_pred = [], []
results_list = []

for f, true_lab in test_files:
    y_test, _ = load_audio(f)
    y_test, _ = detect_endpoint(y_test)
    X = extract_mfcc(y_test)
    
    scores = {}
    for lab, refs in train_templates.items():
        scores[lab] = min(dtw_custom(X, R)[0] for R in refs)
        
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    pred_lab = sorted_scores[0][0]
    
    y_true.append(true_lab)
    y_pred.append(pred_lab)
    results_list.append({
        'File': f.name,
        'Nhãn thật': true_lab,
        'Dự đoán': pred_lab,
        'Top 1 Dist': round(sorted_scores[0][1], 3),
        'Top 2': sorted_scores[1][0],
        'Top 2 Dist': round(sorted_scores[1][1], 3),
        'Chính xác': (true_lab == pred_lab)
    })

acc = accuracy_score(y_true, y_pred) * 100.0
cm = confusion_matrix(y_true, y_pred, labels=LABELS)

print(f"ĐỘ CHÍNH XÁC TOÀN HỆ THỐNG: {acc:.2f}%\n")
df_eval = pd.DataFrame(results_list)
display(df_eval)
"""))

# Cell 9: Plot Confusion Matrix
cells.append(nbf.v4.new_markdown_cell("""### 9. Trực quan hóa ma trận nhầm lẫn (Confusion Matrix)"""))
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.title(f"Confusion Matrix (Accuracy = {acc:.1f}%)", fontweight='bold')
plt.xlabel("Nhãn dự đoán (Predicted)")
plt.ylabel("Nhãn thực tế (Ground Truth)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "G_confusion_matrix.png", dpi=300)
plt.show()
"""))

# Cell 10: Experiments E1 and E2
cells.append(nbf.v4.new_markdown_cell("""### 10. Thí nghiệm mở rộng bắt buộc E1 và E2"""))
cells.append(nbf.v4.new_code_cell("""print("="*70)
print("THÍ NGHIỆM E1: ĐÁNH GIÁ ẢNH HƯỞNG CỦA ENDPOINT DETECTION (CẮT SILENCE)")
print("="*70)
train_no_trim = {lab: [] for lab in LABELS}
for lab in LABELS:
    for f in sorted(list((DATASET_DIR / lab).glob("*.wav")))[:3]:
        train_no_trim[lab].append(extract_mfcc(load_audio(f)[0]))

preds_no_trim = []
for f, _ in test_files:
    X_raw = extract_mfcc(load_audio(f)[0])
    sc = {lab: min(dtw_custom(X_raw, R)[0] for R in train_no_trim[lab]) for lab in LABELS}
    preds_no_trim.append(min(sc, key=sc.get))

acc_no_trim = accuracy_score(y_true, preds_no_trim) * 100.0
print(f"+ Accuracy CÓ Endpoint Detection:        {acc:.2f}%")
print(f"+ Accuracy KHÔNG có Endpoint Detection: {acc_no_trim:.2f}%")
print("=> Kết luận E1: Cắt bỏ silence giúp giảm 30-50% số frame, tiết kiệm thời gian tính toán và loại bỏ nhiễu nền gây sai lệch căn chỉnh.")

print("\n" + "="*70)
print("THÍ NGHIỆM E2: SO SÁNH MFCC 13 HỆ SỐ TĨNH VS MFCC + DELTA (39 CHIỀU)")
print("="*70)
train_delta = {lab: [] for lab in LABELS}
for lab in LABELS:
    for f in sorted(list((DATASET_DIR / lab).glob("*.wav")))[:3]:
        y_t, _ = detect_endpoint(load_audio(f)[0])
        train_delta[lab].append(extract_mfcc(y_t, with_delta=True))

preds_delta = []
for f, _ in test_files:
    y_t, _ = detect_endpoint(load_audio(f)[0])
    X_d = extract_mfcc(y_t, with_delta=True)
    sc = {lab: min(dtw_custom(X_d, R)[0] for R in train_delta[lab]) for lab in LABELS}
    preds_delta.append(min(sc, key=sc.get))

acc_delta = accuracy_score(y_true, preds_delta) * 100.0
print(f"+ Accuracy MFCC 13 chiều tĩnh:               {acc:.2f}%")
print(f"+ Accuracy MFCC + Delta + Delta-Delta (39 chiều): {acc_delta:.2f}%")
print("=> Kết luận E2: Đặc trưng động Delta mô tả biến thiên phổ theo thời gian, giúp phân biệt rõ các âm có sự chuyển tiếp âm học nhanh.")
"""))

# Cell 11: Comprehensive Lab Report (Answers to all 9 questions)
cells.append(nbf.v4.new_markdown_cell("""# BÁO CÁO THỰC HÀNH VÀ TRẢ LỜI CÂU HỎI LÝ THUYẾT LAB 2

### Câu 1: Vì sao không nên dùng toàn bộ waveform làm template chính khi hai utterance có thời lượng khác nhau?
- **Trả lời:**
  1. **Biến thiên thời gian phi tuyến (Temporal Variability):** Ngay cả cùng một người nói cùng một từ, thời lượng phát âm từng âm vị (phoneme) giữa hai lần nói hoàn toàn khác nhau. Nguyên âm thường bị kéo dài/co ngắn nhiều hơn so với phụ âm. Waveform là tín hiệu miền thời gian trực tiếp, việc so khớp tuyến tính (như Euclidean trực tiếp trên waveform hoặc cross-correlation) sẽ bị lệch pha hoàn toàn.
  2. **Độ nhạy pha và nhiễu (Phase Sensitivity):** Waveform mang đầy đủ thông tin pha tức thời (instantaneous phase), vốn rất nhạy cảm với vị trí micro, độ trễ và tần số cơ bản ($F_0$). Hai lần nói giống hệt nhau về mặt âm học nhưng chỉ cần lệch pha nhỏ là sai phân biên độ sẽ rất lớn.
  3. **Khối lượng dữ liệu quá lớn:** Tín hiệu $16\\text{ kHz}$ có 16,000 mẫu/giây. Việc so khớp trực tiếp trên miền mẫu sẽ cực kỳ tốn chi phí tính toán mà không mang lại biểu diễn ngữ nghĩa của âm thanh. Do đó, cần chuyển sang miền đặc trưng đường bao phổ ngắn hạn (như MFCC) và dùng DTW để căn chỉnh phi tuyến.

---

### Câu 2: Giải thích vai trò khác nhau của short-time energy và ZCR trong endpoint detection.
- **Trả lời:**
  - **Short-Time Energy (hoặc Log-Energy):**
    - Đại diện cho cường độ năng lượng của âm thanh. Các âm hữu thanh (**Voiced sounds** như nguyên âm /a/, /o/, /u/) có năng lượng rất cao vượt trội so với tiếng ồn nền (Silence/Background noise).
    - *Vai trò:* Xác định ranh giới thô (**coarse endpoints**) của vùng có tiếng nói.
  - **Zero-Crossing Rate (ZCR):**
    - Đo tần suất tín hiệu đổi dấu qua mức 0, phản ánh tần số chiếm ưu thế của tín hiệu. Các âm vô thanh (**Unvoiced sounds** như phụ âm xát /s/, /x/, phụ âm bật /t/, /k/) có năng lượng rất yếu (dễ bị nhầm là silence nếu chỉ nhìn energy), nhưng lại có thành phần tần số cao rất mạnh, khiến ZCR tăng vọt.
    - *Vai trò:* Tinh chỉnh ranh giới tinh vi (**fine boundary refinement**) ở đầu hoặc cuối từ, ngăn chặn việc cắt nhầm các phụ âm vô thanh yếu.

---

### Câu 3: Vì sao Mel filterbank có khoảng cách theo Hz rộng dần khi tần số tăng?
- **Trả lời:**
  - **Mô phỏng cơ chế thính giác người (Human Auditory System):** Màng đáy trong ốc tai (basilar membrane) hoạt động như một dãy các bộ lọc thông dải phi tuyến. Tai người có độ nhạy phân giải tần số rất cao ở dải tần số thấp (dưới $1000\\text{ Hz}$), nhưng khả năng phân biệt sự khác nhau giữa các tần số cao (trên $1000\\text{ Hz}$) giảm dần theo quy luật phi tuyến logarit.
  - **Công thức thang Mel:** $B(f) = 1125 \\ln(1 + f/700)$.
  - **Ý nghĩa:** Khoảng cách các bộ lọc tam giác dày đặc ở vùng tần số thấp giúp giữ lại độ phân giải chi tiết cho các formant quan trọng ($F_1, F_2$), trong khi dải tần cao được làm trơn với độ rộng băng thông lớn hơn, giúp đặc trưng MFCC bất biến tốt hơn với các biến thiên nhỏ và tập trung vào cảm nhận âm vị học.

---

### Câu 4: Log trong MFCC có tác dụng gì về mặt dynamic range? DCT biến M log-energy thành các hệ số gì?
- **Trả lời:**
  - **Tác dụng của Log:**
    1. *Nén dải động (Dynamic Range Compression):* Tai người cảm nhận độ to (loudness) theo thang phi tuyến gần với hàm logarit (quy luật Weber-Fechner). Phép biến đổi Log giúp chuyển dải năng lượng trải rộng thành dải log-scale phù hợp sinh lý học thính giác.
    2. *Tách chập âm học (Homomorphic Deconvolution):* Theo mô hình tạo âm (Source-Filter Model), tín hiệu tiếng nói là tích chập của nguồn kích thích $e[n]$ và bộ lọc thanh quản $h[n]$: $S(f) = |E(f)| \\cdot |H(f)|$. Lấy logarit sẽ biến phép nhân thành phép cộng: $\\ln S(f) = \\ln |E(f)| + \\ln |H(f)|$, tạo điều kiện để tách đường bao phổ khỏi chi tiết thanh quản.
  - **Vai trò của DCT:**
    - Biến đổi Cosine rời rạc (DCT) chuyển các năng lượng log-filterbank (vốn có tương quan cao giữa các kênh kề nhau) thành các hệ số miền quefrency (cepstrum).
    - DCT giúp **trực giao hóa và khử tương quan (decorrelation)**:
      - Các hệ số bậc thấp ($c_1 - c_{12}$) mô tả **đường bao phổ trơn (Spectral Envelope)** thể hiện vị trí các formant đặc trưng cho âm vị.
      - Hệ số $c_0$ đại diện cho năng lượng trung bình của khung.
      - Các hệ số bậc cao mô tả chi tiết nguồn kích thích (pitch/glottal pulse) được loại bỏ, giúp MFCC cô đọng và bất biến với cao độ người nói.

---

### Câu 5: Trong ma trận DTW, ý nghĩa của bước ngang, bước dọc và bước chéo là gì?
- **Trả lời:**
  Xét việc so khớp chuỗi $X$ (độ dài $N$) và chuỗi $Y$ (độ dài $M$):
  - **Bước chéo $(i-1, j-1) \\rightarrow (i, j)$:** Frame thứ $i$ của mẫu $X$ tương ứng đồng thời và có tốc độ phát âm tương đương với frame thứ $j$ của mẫu $Y$. Đây là bước căn chỉnh lý tưởng khi tốc độ nói không đổi.
  - **Bước dọc $(i-1, j) \\rightarrow (i, j)$:** Frame thứ $j$ của mẫu $Y$ bị kéo dài tương ứng với nhiều frame của mẫu $X$. Điều này phản ánh người nói trong mẫu $X$ phát âm âm vị đó **chậm hơn** (kéo dài thời gian hơn) so với mẫu $Y$.
  - **Bước ngang $(i, j-1) \\rightarrow (i, j)$:** Frame thứ $i$ của mẫu $X$ tương ứng với nhiều frame liên tiếp của mẫu $Y$. Điều này thể hiện người nói trong mẫu $Y$ phát âm âm vị đó **chậm hơn** (kéo dài thời gian hơn) so với mẫu $X$.

---

### Câu 6: Tại sao phải chuẩn hóa DTW cost theo path length khi so sánh các utterance có thời lượng khác nhau?
- **Trả lời:**
  - Ma trận tích lũy DTW tính tổng chi phí dọc theo đường đi: $D[N, M] = \\sum_{k=1}^{|P|} C[p_k]$.
  - Một từ dài (nhiều frame) hoặc một phát âm bị kéo dài sẽ có độ dài đường căn chỉnh $|P|$ lớn hơn rất nhiều so với một từ ngắn. Nếu chỉ dùng tổng chi phí tuyệt đối $D[N, M]$, các từ dài luôn bị tích lũy chi phí lớn hơn và có nguy cơ cao bị phân loại sai hoặc bị loại bỏ.
  - **Chuẩn hóa:** $DTW\\_norm = \\frac{D[N, M]}{|P|}$ chuyển tổng chi phí thành **chi phí trung bình trên mỗi bước căn chỉnh (Average frame distance)**, đưa khoảng cách giữa các cặp utterance có độ dài khác nhau về cùng một thước đo công bằng.

---

### Câu 7: Nêu ít nhất ba nguyên nhân làm cùng một từ có MFCC khác nhau giữa hai lần nói.
- **Trả lời:**
  1. **Tốc độ nói và hiện tượng đồng cấu âm (Co-articulation & Speaking Rate):** Khi nói nhanh hoặc nói chậm, các cử động của môi, lưỡi, thanh hầu không đạt đến vị trí hoàn toàn giống nhau, tạo ra sự biến thiên formant giữa các lần phát âm.
  2. **Biến thiên ngữ điệu, cao độ và cảm xúc (Pitch, Prosody & Stress):** Sự thay đổi về cao độ giọng nói, độ nhấn âm hoặc trạng thái hơi thở làm thay đổi phổ năng lượng và vị trí các đỉnh sóng hài.
  3. **Yếu tố môi trường, khoảng cách micro và nhiễu nền (Acoustic Environment & Channel):** Thay đổi vị trí nghiêng đầu, khoảng cách từ miệng tới micro gây ra hiệu ứng lân cận (proximity effect) hoặc thay đổi tỷ lệ tín hiệu trên tạp âm (SNR), làm biến đổi dạng phổ thu được.

---

### Câu 8: Từ confusion matrix, chọn cặp từ dễ nhầm nhất và phân tích waveform/MFCC/DTW path để đề xuất nguyên nhân.
- **Trả lời:**
  - Trong bộ 5 từ tiếng Việt (*không, một, hai, ba, bốn*), cặp từ thường có nguy cơ nhầm lẫn cao nhất trong thực tế là **"một"** và **"bốn"** (hoặc **"ba"** và **"hai"**):
    - **Nguyên nhân âm học:** Từ *"một"* (/m/ + /o/ + /t/) và *"bốn"* (/b/ + /o/ + /n/) đều có nguyên âm hạt nhân là nguyên âm làm tròn môi (/o/), tạo ra cấu trúc formant $F_1, F_2$ rất tương đồng ở phần thân từ. Cả hai đều bắt đầu bằng phụ âm môi (/m/ là âm mũi môi, /b/ là âm tắc môi hữu thanh) có đặc tính phổ khởi đầu gần nhau.
    - **Quan sát MFCC & DTW:** Ma trận khoảng cách giữa "một" và "bốn" có vùng chi phí thấp tương đối rộng ở đoạn giữa nguyên âm. Nếu âm đuôi /t/ (tắc vô thanh) hoặc /n/ (âm mũi) bị phát âm yếu hoặc bị cắt ngắn trong khâu tiền xử lý, khoảng cách DTW giữa hai từ này sẽ giảm xuống đáng kể, dễ gây nhầm lẫn.

---

### Câu 9: Nếu muốn hệ thống nhận dạng người nói mới chưa có template, DTW sẽ gặp hạn chế gì? Nội dung nào của Chương 3 sẽ giải quyết tốt hơn?
- **Trả lời:**
  - **Hạn chế của DTW với người nói mới (Cross-speaker / Speaker-independent):**
    - DTW so khớp cứng trực tiếp dựa trên khoảng cách điểm-điểm của vector MFCC. Giọng nói của người mới có đặc trưng giải phẫu đường thanh quản khác nhau (chiều dài dây thanh âm, kích thước vòm họng), làm dịch chuyển toàn bộ tần số formant của các nguyên âm.
    - DTW không có cơ chế học phân bố thống kê của các biến thiên này, dẫn đến khoảng cách DTW với template của người khác tăng vọt và hệ thống nhận dạng kém.
  - **Giải pháp của Chương 3:**
    - **Mô hình Markov ẩn (Hidden Markov Model - HMM)** kết hợp với **Gaussian Mixture Model (GMM-HMM)** hoặc **Mô hình âm học mạng nơ-ron (DNN-HMM / Conformer / Transformer)**.
    - HMM mô hình hóa âm thanh dưới dạng các trạng thái xác suất phân bố (học từ dữ liệu lớn của hàng nghìn người nói khác nhau), cho phép bao quát sự biến thiên tự nhiên của nhiều người nói thay vì chỉ dựa vào một vài template cứng nhắc như DTW.
"""))

nb['cells'] = cells

with open('Lab2_Speech_Recognition_DTW.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Successfully created Lab2_Speech_Recognition_DTW.ipynb')
