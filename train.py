import os
# 0) TF C++ 로그 숨기기 (반드시 tensorflow import 이전)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 0.1) matplotlib 비대화형 백엔드로 설정 (plt.show() 블록 방지)
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from tensorflow.keras.utils import to_categorical

# ---------- 오디오 → 3초 세그먼트 ----------
def trim_audio(audio_path, sr=22_050, duration=3):
    print(f"    ▶ trim_audio: loading {audio_path.name}")
    y, _ = librosa.load(audio_path, sr=sr)
    seg_len = sr * duration
    return [y[i:i+seg_len] for i in range(0, len(y) - seg_len + 1, seg_len)], sr

# ---------- 세그먼트 → PNG 스펙트로그램 ----------
def save_spectrogram(segment, sr, save_path):
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(4, 4), dpi=100)
    librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# ---------- 데이터셋 생성 ----------
def generate_spectrograms(base_dir: Path, out_dir: Path):
    print("▶ [1] generate_spectrograms 시작")
    for label_dir in sorted(d for d in base_dir.iterdir() if d.is_dir()):
        print(f"    ▶ 레이블 디렉토리: {label_dir.name}")
        for ext in ("*.mp3", "*.wav"):
            for audio in label_dir.glob(ext):
                segments, sr = trim_audio(audio)
                for i, seg in enumerate(segments):
                    out_file = out_dir / label_dir.name / f"{audio.stem}_{i}.png"
                    save_spectrogram(seg, sr, out_file)
    print("▶ [1] generate_spectrograms 완료")

def load_dataset(image_dir: Path):
    print("▶ [2] load_dataset 시작")
    labels = sorted(d.name for d in image_dir.iterdir() if d.is_dir())
    label_map = {lbl: idx for idx, lbl in enumerate(labels)}

    images, targets = [], []
    for lbl in labels:
        for img_path in (image_dir / lbl).glob("*.png"):
            img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
            arr = tf.keras.utils.img_to_array(img) / 255.0
            images.append(arr)
            targets.append(label_map[lbl])
    print("▶ [2] load_dataset 완료")
    return np.array(images), np.array(targets), label_map

# ---------- CNN 모델 정의 ----------
def build_cnn(input_shape, n_cls):
    l2 = 1e-4
    return models.Sequential([
        layers.Conv2D(32, 3, padding='same',
                      kernel_regularizer=regularizers.l2(l2),
                      input_shape=input_shape),
        layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same',
                      kernel_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same',
                      kernel_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2)),
        layers.Dropout(0.5),
        layers.Dense(n_cls, activation='softmax')
    ])

# ---------- 메인 ----------
if __name__ == "__main__":
    print("▶ [0] 스크립트 시작")
    base_dir = Path("data")
    spec_dir = Path("spectrograms")

    # 1) 스펙트로그램 생성
    generate_spectrograms(base_dir, spec_dir)

    # 2) 데이터 로드
    X, y, label_map = load_dataset(spec_dir)
    print("▶ Label map:", label_map)

    # 3) 학습/검증 분할
    print("▶ [3] train_test_split")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("▶ [3] split 완료")

    # 4) 모델 빌드 & 컴파일
    print("▶ [4] 모델 빌드")
    model = build_cnn((128, 128, 3), n_cls=len(label_map))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("▶ [4] 컴파일 완료")

    # 5) EarlyStopping 콜백
    es = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # 6) 학습
    print("▶ [5] 학습 시작")
    history = model.fit(
        X_tr, y_tr,
        epochs=50,
        batch_size=32,
        validation_data=(X_te, y_te),
        shuffle=True,
        callbacks=[es]
    )
    print("▶ [5] 학습 완료")

    # 7) 모델 저장
    print("▶ [6] 모델 저장")
    model.save("bee_hornet_cnn.h5")

    # ======================
    # 8) 성능 지표 저장
    # ======================
    print("▶ [7] 성능 지표 생성")
    y_prob = model.predict(X_te)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = y_te
    labels = list(label_map.keys())
    n_cls = len(labels)

    # 8-2) 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, aspect='equal')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    ticks = np.arange(n_cls)
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # 8-3) 분류 리포트
    print("\n▶ Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels))

    # 8-4) 클래스별 IoU
    print("▶ Class-wise IoU:")
    for i, lbl in enumerate(labels):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        print(f"  {lbl}: {iou:.3f}")

    # 8-5) ROC 곡선
    y_true_ohe = to_categorical(y_true, num_classes=n_cls)
    plt.figure()
    for i, lbl in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_ohe[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lbl} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', label="Chance")
    plt.title("Multi-class ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

    print("▶ [7] 성능 지표 저장 완료 (confusion_matrix.png, roc_curve.png)")
    print("▶ 스크립트 종료")
