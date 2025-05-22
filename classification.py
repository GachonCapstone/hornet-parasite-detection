import os
from pathlib import Path
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ---------- 오디오 → 3초 세그먼트 ----------
def trim_audio(audio_path, sr=22_050, duration=3):
    y, _ = librosa.load(audio_path, sr=sr)
    seg_len = sr * duration
    return [y[i:i+seg_len] for i in range(0, len(y) - seg_len + 1, seg_len)], sr

# ---------- 세그먼트 → PNG 스펙트로그램 ----------
def save_spectrogram(segment, sr, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 4), dpi=100)
    librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# ---------- 데이터셋 준비 ----------
def generate_spectrograms(base_dir: Path, out_dir: Path):
    for label_dir in sorted(d for d in base_dir.iterdir() if d.is_dir()):
        # 처리할 오디오 확장자
        for ext in ("*.mp3", "*.wav"):
            for audio in label_dir.glob(ext):
                segments, sr = trim_audio(audio)
                for i, seg in enumerate(segments):
                    out_file = out_dir / label_dir.name / f"{audio.stem}_{i}.png"
                    save_spectrogram(seg, sr, out_file)

def load_dataset(image_dir: Path):
    labels = sorted(d.name for d in image_dir.iterdir() if d.is_dir())
    label_map = {lbl: idx for idx, lbl in enumerate(labels)}

    images, targets = [], []
    for lbl in labels:
        for img_path in (image_dir / lbl).glob("*.png"):
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            images.append(tf.keras.utils.img_to_array(img) / 255.0)
            targets.append(label_map[lbl])

    return np.array(images), np.array(targets), label_map

# ---------- CNN 모델 정의 ----------
def build_cnn(input_shape, n_cls):
    return models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_cls, activation='softmax')
    ])

# ---------- 메인 ----------
if __name__ == "__main__":
    base_dir = Path("data")            # 클래스별 .mp3/.wav 파일 디렉토리
    spec_dir = Path("spectrograms")    # 생성된 스펙트로그램 저장 디렉토리

    print("GPUs:", tf.config.list_physical_devices('GPU'))

    # 1) 스펙트로그램 생성
    generate_spectrograms(base_dir, spec_dir)

    # 2) 데이터 로드
    X, y, label_map = load_dataset(spec_dir)
    print("Label map:", label_map)

    # 3) 학습/검증 분할
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4) 모델 빌드 & 컴파일
    model = build_cnn((128, 128, 3), n_cls=len(label_map))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5) 학습
    model.fit(X_tr, y_tr, epochs=20, validation_data=(X_te, y_te))

    # 6) 모델 저장
    model.save("bee_hornet_cnn.h5")
