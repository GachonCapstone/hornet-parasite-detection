import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1. 스펙트로그램 저장 함수
def save_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=22050)
    plt.figure(figsize=(4, 4))  # 정사각형 이미지로 저장
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 2. 데이터셋 폴더 순회하며 스펙트로그램 생성
def generate_spectrograms(base_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)

        for file in os.listdir(label_dir):
            if file.endswith('.mp3'):
                input_path = os.path.join(label_dir, file)
                output_path = os.path.join(output_label_dir, file.replace('.mp3', '.png'))
                save_spectrogram(input_path, output_path)

# 3. 스펙트로그램 이미지 로드 및 레이블링
def load_dataset(image_dir):
    images = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(os.listdir(image_dir))}

    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        for file in os.listdir(label_dir):
            if file.endswith('.png'):
                img_path = os.path.join(label_dir, file)
                img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
                img_array = tf.keras.utils.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label_map[label])

    return np.array(images), np.array(labels), label_map

# 4. CNN 모델 정의
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 메인 실행 코드
if __name__ == "__main__":
    base_dir = 'data'
    spectrogram_dir = 'spectrograms'

    # gpu 인식 여부 확인
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # 1. MP3 → 스펙트로그램 PNG 생성
    generate_spectrograms(base_dir, spectrogram_dir)

    # 2. 스펙트로그램 이미지 데이터 로드
    images, labels, label_map = load_dataset(spectrogram_dir)
    print(f"Label Map: {label_map}")

    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # 4. CNN 모델 학습
    model = build_cnn_model(input_shape=(128, 128, 3), num_classes=len(label_map))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    # 5. 학습 결과 저장 (옵션)
    model.save('bee_vs_hornet_cnn.h5')

