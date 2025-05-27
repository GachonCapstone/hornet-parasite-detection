import io
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

def predict_audio_bytes(
    audio_bytes: bytes,
    model_path: str,
    duration: int = 3,
    sr: int = 22050
) -> list[dict]:
    """
    바이트 스트림으로 전달된 오디오 데이터를 지정된 duration(초) 단위로 분할하여
    모델로 추론한 결과를 JSON-serializable 리스트 형태로 반환

    Parameters:
    - audio_bytes: 오디오 파일의 바이트 스트림 (e.g., .mp3, .wav 데이터)
    - model_path: 학습된 Keras 모델 파일 경로 (.h5)
    - duration: 분할 단위 길이(초)
    - sr: 샘플링 레이트

    Returns:
    - results: [
          {
            'segment_index': int,
            'predicted_label': str,
            'confidence': float,
            'raw_probabilities': [float,...]
          },
          ...
      ]
    """
    # 1) 오디오 데이터를 바이트 스트림에서 로드
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)

    # 2) 모델 로드
    model = load_model(model_path)
    label_map = {0: 'honeybee', 1: 'hornet'}

    # 3) 오디오 분할
    segment_length = sr * duration
    segments = [
        y[i:i+segment_length]
        for i in range(0, len(y), segment_length)
        if len(y[i:i+segment_length]) == segment_length
    ]

    results = []
    # 4) 각 세그먼트 스펙트로그램 생성 및 예측
    for idx, segment in enumerate(segments):
        # Mel-spectrogram 계산
        S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        # 0~1 정규화
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)

        # 채널 및 배치 차원 추가
        img = S_norm[..., np.newaxis]  # (128, T, 1)
        img_resized = tf.image.resize(img, [128, 128]).numpy()  # (128,128,1)
        img_3ch = np.repeat(img_resized, 3, axis=-1)  # (128,128,3)
        input_batch = np.expand_dims(img_3ch, axis=0)  # (1,128,128,3)

        # 예측 실행
        preds = model.predict(input_batch, verbose=0)[0]
        cls = int(np.argmax(preds))
        results.append({
            'segment_index': idx,
            'predicted_label': label_map.get(cls, str(cls)),
            'confidence': float(preds[cls]),
            'raw_probabilities': [float(p) for p in preds]
        })

    return results