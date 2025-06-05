import logging
import os
import cv2
import numpy as np
from ultralytics import YOLO

# 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "parasite.pt")

NAME     = "parasite"
MODEL    = YOLO(MODEL_PATH)  # 1회만 로드
CLS_ID   = {1, 5}
CONF_THR = 0.5

def parasite_detection(image_bytes: bytes) -> dict:
    """
    image_bytes: 바이트 스트림으로 전달된 이미지 데이터 (.jpg, .png 등)
    반환값: {'count': int, 'score': float}
    """
    try:
        # 1) 바이트 스트림을 NumPy 배열로 변환, OpenCV 디코딩
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"count": 0, "score": 0.0}

        # 2) YOLO 모델 추론
        res = MODEL(frame)[0]
        confs = [
            float(b.conf)
            for b in res.boxes
            if int(b.cls) in CLS_ID and float(b.conf) >= CONF_THR
        ]

        return {
            "count": len(confs),
            "score": round(max(confs), 3) if confs else 0.0
        }

    except Exception:
        logging.exception("기생충 탐지 실패")
        return {"count": 0, "score": 0.0}