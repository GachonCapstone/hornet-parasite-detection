import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 스펙트로그램 저장 함수 (학습 때 썼던 거 재활용)
def save_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=22050)
    plt.figure(figsize=(4, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 스펙트로그램 이미지 → 모델 입력용 배열 변환
def load_spectrogram_as_input(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # 학습 때 썼던 사이즈
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # 배치 차원 추가

# 모델 로드
model = load_model('bee_vs_hornet_cnn.h5')

# 레이블 매핑 (학습할 때의 레이블 순서랑 동일해야 함)
label_map = {0: 'honeybee', 1: 'hornet'}

# 테스트 데이터 경로
test_data_dir = 'test_data'
temp_spectrogram_dir = 'temp_spectrograms'

# 임시 스펙트로그램 저장 폴더
os.makedirs(temp_spectrogram_dir, exist_ok=True)

# 테스트 데이터 순회하며 예측
for label in ['honeybee', 'hornet']:
    label_dir = os.path.join(test_data_dir, label)

    for file_name in os.listdir(label_dir):
        if not file_name.endswith('.mp3'):
            continue

        mp3_path = os.path.join(label_dir, file_name)
        spectrogram_path = os.path.join(temp_spectrogram_dir, f'{label}_{file_name.replace(".mp3", ".png")}')

        # MP3 → 스펙트로그램 변환
        save_spectrogram(mp3_path, spectrogram_path)

        # 스펙트로그램 → 모델 입력 데이터
        input_data = load_spectrogram_as_input(spectrogram_path)

        # 예측
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        predicted_label = label_map[predicted_class]

        print(f'{file_name}: 실제={label}, 예측={predicted_label}, 확률={prediction[0][predicted_class]:.2f}')
