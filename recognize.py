"""
================================================================================
SCRIPT NHẬN DẠNG FILE ÂM THANH BẤT KỲ DÙNG DTW (LAB 2 - CSE457)
================================================================================
Hướng dẫn sử dụng:
  Cách 1 (Chạy qua dòng lệnh Terminal):
      python recognize.py 1.mp3
      python recognize.py path/to/your_audio.wav

  Cách 2 (Chạy trực tiếp trong Python / Jupyter Notebook):
      from recognize import recognize_single_file
      pred, top3 = recognize_single_file("1.mp3")
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import librosa
import scipy.signal as signal

# Đảm bảo hiển thị tiếng Việt trên terminal Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 1. Cấu hình các tham số chuẩn của Lab 2
FS = 16000             # Tần số lấy mẫu: 16 kHz
FRAME_MS = 25          # Khung: 25 ms
HOP_MS = 10            # Bước dịch: 10 ms
WIN = int(FS * FRAME_MS / 1000)  # L = 400 mẫu
HOP = int(FS * HOP_MS / 1000)    # R = 160 mẫu
N_FFT = 512
N_MELS = 24
N_MFCC = 13
PRE_EMPH = 0.97

# Ngưỡng loại bỏ từ lạ (Rejection threshold theta) theo mục F của đề Lab
# Nếu khoảng cách DTW nhỏ nhất > REJECT_THRESHOLD => coi là từ lạ (unknown)
REJECT_THRESHOLD = 25.0

LABELS = ['khong', 'mot', 'hai', 'ba', 'bon']
LABEL_NAMES = {
    'khong': 'Không (0)',
    'mot': 'Một (1)',
    'hai': 'Hai (2)',
    'ba': 'Ba (3)',
    'bon': 'Bốn (4)'
}


def load_and_preprocess_audio(audio_path):
    """
    Đọc file âm thanh bất kỳ (MP3, WAV, OGG, M4A...), tự động chuyển về 16kHz Mono
    và chuẩn hóa biên độ đỉnh để chống biến thiên âm lượng.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {audio_path}")
        
    y, sr = librosa.load(str(path), sr=FS, mono=True)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / (max_val + 1e-9)
    return y, sr


def trim_silence(y, sr=FS, top_db=32, margin_ms=50):
    """
    Cắt bỏ khoảng lặng đầu/cuối (Endpoint Detection) có chèn thêm margin an toàn.
    """
    yt, idx = librosa.effects.trim(y, top_db=top_db, frame_length=WIN, hop_length=HOP)
    margin_samples = int(sr * margin_ms / 1000)
    s = max(0, idx[0] - margin_samples)
    e = min(len(y), idx[1] + margin_samples)
    return y[s:e]


def extract_mfcc(y, sr=FS):
    """
    Trích xuất 13 hệ số MFCC với bộ lọc tiền nhấn Pre-emphasis và chuẩn hóa CMN.
    """
    # Bộ lọc tiền nhấn
    y_pre = signal.lfilter([1.0, -PRE_EMPH], [1.0], y)
    
    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y_pre, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS,
        n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        window='hamming', center=False
    )
    
    # Chuẩn hóa trung bình phổ (Cepstral Mean Normalization)
    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
    return mfcc.T  # Kích thước: (T_frames, 13)


def dtw_distance(X, Y):
    """
    Thuật toán Dynamic Time Warping (DTW) tự cài đặt bằng quy hoạch động:
    - Khoảng cách cục bộ Euclidean
    - Ma trận tích lũy chi phí D[i, j]
    - Chuẩn hóa theo độ dài đường căn chỉnh |P|
    """
    N, M = len(X), len(Y)
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    C = np.linalg.norm(diff, axis=2)
    
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    back = np.zeros((N + 1, M + 1, 2), dtype=int)
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            local_cost = C[i - 1, j - 1]
            choices = [
                (D[i - 1, j - 1], i - 1, j - 1),
                (D[i - 1, j],     i - 1, j),
                (D[i, j - 1],     i,     j - 1)
            ]
            best, pi, pj = min(choices, key=lambda z: z[0])
            D[i, j] = local_cost + best
            back[i, j] = [pi, pj]
            
    # Backtrack
    path_len = 0
    curr_i, curr_j = N, M
    while curr_i > 0 and curr_j > 0:
        path_len += 1
        curr_i, curr_j = back[curr_i, curr_j]
        
    dtw_norm = D[N, M] / max(path_len, 1)
    return dtw_norm


# Bộ nhớ đệm lưu trữ templates để không phải tính lại nhiều lần
_CACHED_TEMPLATES = None

def load_or_build_templates(dataset_dir="dataset"):
    """
    Đọc các file mẫu trong thư mục dataset để tạo cơ sở dữ liệu Template MFCC.
    Mặc định lấy 3 file đầu tiên của mỗi từ làm reference templates.
    """
    global _CACHED_TEMPLATES
    if _CACHED_TEMPLATES is not None:
        return _CACHED_TEMPLATES
        
    data_path = Path(dataset_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục template: {dataset_dir}. Hãy chạy `python lab2_main.py` trước!")
        
    templates = {lab: [] for lab in LABELS}
    for lab in LABELS:
        wav_files = sorted(list((data_path / lab).glob("*.wav")))[:3]
        for f in wav_files:
            y, _ = load_and_preprocess_audio(f)
            y = trim_silence(y)
            feat = extract_mfcc(y)
            templates[lab].append(feat)
            
    _CACHED_TEMPLATES = templates
    return templates


def recognize_single_file(audio_path, reject_threshold=REJECT_THRESHOLD, dataset_dir="dataset"):
    """
    Hàm nhận dạng chính:
    - Đầu vào: đường dẫn file âm thanh bất kỳ (mp3, wav, ...)
    - Trả về: 
        best_label: Nhãn dự đoán (hoặc 'unknown' nếu vượt ngưỡng reject)
        sorted_scores: Danh sách (nhãn, khoảng cách DTW) được xếp tăng dần
    """
    templates = load_or_build_templates(dataset_dir)
    
    # 1. Tải và xử lý file âm thanh đầu vào
    y, sr = load_and_preprocess_audio(audio_path)
    dur_raw = len(y) / sr
    y = trim_silence(y)
    dur_trim = len(y) / sr
    
    # 2. Trích xuất MFCC
    X = extract_mfcc(y)
    
    # 3. So khớp DTW với toàn bộ templates
    scores = {}
    for lab, refs in templates.items():
        min_cost = min(dtw_distance(X, R) for R in refs)
        scores[lab] = min_cost
        
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1])
    top_label, top_cost = sorted_scores[0]
    
    # 4. Kiểm tra ngưỡng Reject theo mục F
    if reject_threshold is not None and top_cost > reject_threshold:
        predicted = "unknown"
    else:
        predicted = top_label
        
    return predicted, sorted_scores, (dur_raw, dur_trim)


if __name__ == "__main__":
    # Nếu truyền file qua dòng lệnh: python recognize.py <file_path>
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Nếu không truyền tham số, cho người dùng nhập hoặc dùng file mặc định
        user_input = input("Nhập đường dẫn file âm thanh cần nhận dạng (VD: 1.mp3, 2.wav): ").strip()
        target_file = user_input if user_input else "1.mp3"
        
    print("\n" + "="*70)
    print(f"[*] ĐANG XỬ LÝ VÀ NHẬN DẠNG FILE: {target_file}")
    print("="*70)
    
    try:
        pred_label, scores, (dur_raw, dur_trim) = recognize_single_file(target_file)
        
        print(f"[+] Thời lượng file gốc:  {dur_raw:.2f}s")
        print(f"[+] Thời lượng sau cắt:   {dur_trim:.2f}s (đã lọc silence)")
        print("\n[+] Bảng xếp hạng khoảng cách DTW tới các từ mẫu (Càng nhỏ càng giống):")
        print(f"    {'Xếp hạng':<10} | {'Từ vựng':<15} | {'Ý nghĩa':<12} | {'Khoảng cách DTW':<15}")
        print("    " + "-"*60)
        for rank, (lab, dist) in enumerate(scores, 1):
            vn_name = LABEL_NAMES.get(lab, lab)
            mark = " <=== (Trùng khớp nhất)" if rank == 1 else ""
            print(f"    Top {rank:<6} | {lab:<15} | {vn_name:<12} | {dist:<15.3f}{mark}")
            
        print("\n" + "="*70)
        if pred_label == "unknown":
            print(f"[>>>] KẾT QUẢ DỰ ĐOÁN: TỪ LẠ KHÔNG XÁC ĐỊNH ('unknown')")
            print(f"      (Khoảng cách gần nhất {scores[0][1]:.2f} vượt ngưỡng an toàn θ = {REJECT_THRESHOLD})")
        else:
            vn_text = LABEL_NAMES.get(pred_label, pred_label)
            print(f"[>>>] KẾT QUẢ DỰ ĐOÁN: '{pred_label.upper()}' ({vn_text})")
            print(f"      Độ tin cậy: Khoảng cách DTW tốt nhất = {scores[0][1]:.3f}")
        print("="*70)
        
    except Exception as e:
        print(f"[-] Lỗi trong quá trình nhận dạng: {e}")
