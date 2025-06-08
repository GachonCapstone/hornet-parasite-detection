import cv2
import numpy as np
from ultralytics import YOLO
import os

# YOLO 모델 로드
# 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "image_detection_model.pt")
yolo_model = YOLO(MODEL_PATH)


def infer_image(image_bytes: bytes,
                conf_threshold: float = 0.5,
                hornet_classes: list[int] = [0, 1, 2]) -> list[dict]:
    """
    바이트 스트림으로 입력받은 이미지를 YOLO 검출 후 말벌/꿀벌 확률을 계산하여
    지정된 JSON 형식으로 반환합니다.

    Args:
        image_bytes (bytes): 이미지 데이터를 바이트 스트림으로 입력.
        conf_threshold (float): 검출 신뢰도 임계값.
        hornet_classes (list[int]): 말벌 클래스 인덱스 리스트.

    Returns:
        List[dict]: 다음 형식의 딕셔너리를 요소로 갖는 리스트
        [
          {
            'segment_index': 0,
            'predicted_label': str,
            'confidence': float,
            'raw_probabilities': [honeybee_prob, hornet_prob]
          }
        ]
    """

    hornet_count = 0 # 말벌 개체 수
    hornet_prob_sum = 0.0 # 확률 누적 총합

    # 바이트 스트림을 OpenCV 이미지로 디코딩
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("유효한 이미지 데이터가 아닙니다.")

    # YOLO 검출 수행
    results = yolo_model(frame)
    hornet_prob = 0.0
    for result in results:
        for box in result.boxes:
            conf = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())
            if conf >= conf_threshold and cls in hornet_classes:
                hornet_prob_sum += conf
                hornet_count += 1

    # 평균 확률 계산
    hornet_prob = hornet_prob_sum / hornet_count if hornet_count > 0 else 0.0

    # 꿀벌 확률
    honeybee_prob = 1.0 - hornet_prob
    raw_probs = [honeybee_prob, hornet_prob]

    # 예측 레이블 결정
    idx = int(honeybee_prob < hornet_prob)
    label_map = {0: 'honeybee', 1: 'hornet'}
    predicted_label = label_map[idx]
    confidence = raw_probs[idx]

    # 결과 포맷
    return [{
        'segment_index': 0,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'raw_probabilities': raw_probs,
        'count': hornet_count
    }]
