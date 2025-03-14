import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 오디오 파일을 10초 단위로 자르는 함수
def trim_audio(audio_path, sr=22050, duration=10):
    y, _ = librosa.load(audio_path, sr=sr)
    segment_length = sr * duration  # 10초 길이 샘플 수
    segments = []
    
    for start in range(0, len(y), segment_length):
        segment = y[start:start + segment_length]
        if len(segment) == segment_length:
            segments.append(segment)
    
    return segments, sr

# 스펙트로그램 저장 함수
def save_spectrogram(audio_segment, sr, save_path):
    plt.figure(figsize=(4, 4))
    S = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 스펙트로그램 이미지 → 모델 입력용 배열 변환
def load_spectrogram_as_input(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# 모델 로드
model = load_model('bee_vs_hornet_cnn.h5')

# 레이블 매핑
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
        segments, sr = trim_audio(mp3_path)
        
        for i, segment in enumerate(segments):
            spectrogram_path = os.path.join(temp_spectrogram_dir, f'{label}_{file_name.replace(".mp3", f"_{i}.png")}')
            
            # MP3 → 스펙트로그램 변환
            save_spectrogram(segment, sr, spectrogram_path)

            # 스펙트로그램 → 모델 입력 데이터
            input_data = load_spectrogram_as_input(spectrogram_path)

            # 예측
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            predicted_label = label_map[predicted_class]

            print(f'{file_name} (Segment {i}): 실제={label}, 예측={predicted_label}, 확률={prediction[0][predicted_class]:.2f}')
