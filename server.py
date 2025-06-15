import json
import base64
import threading
import time
import queue

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import requests

from hornet_audio_detection import predict_audio_bytes
from hornet_image_detection import infer_image
from parasite_image_detection import parasite_detection
from datetime import datetime

# 요청 대기 큐
inference_queue = queue.Queue()

# 메시지 수신 시 실행될 콜백
def on_message(client, userdata, msg):
    print(f"[Receive] Topic: {msg.topic}")
    inference_queue.put((msg.topic, msg.payload))

# late fusion 함수 unchanged
def late_fusion(audio_results, image_results,
                w_audio: float = 0.4, w_image: float = 0.6) -> str:
    a = audio_results[0]
    i = image_results[0]

    a_score = a.get('confidence', 0.0) * w_audio
    i_score = i.get('confidence', 0.0) * w_image

    return a['predicted_label'] if a_score > i_score else i['predicted_label']


def inference_worker():
    print("[Worker] Inference worker started.")
    while True:
        topic, payload = inference_queue.get()
        try:
            data = json.loads(payload.decode('utf-8'))
            hive_id = data.get('id')
            audio_b64 = data.get('audio')
            image_b64 = data.get('image')

            if not audio_b64 or not image_b64:
                print("[Worker] Error: 'audio' or 'image' key missing")
            else:
                # 1) 디코딩 & 추론
                audio_bytes = base64.b64decode(audio_b64)
                audio_results = predict_audio_bytes(audio_bytes=audio_bytes)
                print(f"[Worker] Audio probs: {audio_results}")

                image_bytes = base64.b64decode(image_b64)
                image_results = infer_image(
                    image_bytes=image_bytes,
                    conf_threshold=0.5
                )
                print(f"[Worker] Image probs: {image_results}")

                parasite_image_results = parasite_detection(image_bytes)
                print(f"[Worker] Parasite Image probs: {parasite_image_results}")

                # 2) 이미지 디코딩
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # 박스 및 라벨 렌더링
                for det in image_results:
                    bbox = det.get('bbox')  # [xmin, ymin, xmax, ymax]
                    label = det.get('label', '')
                    score = det.get('confidence', 0)
                    if bbox:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"{label}:{score:.2f}"
                        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1)

                # 호넷 개수 표시
                hornet_count = sum(det.get('count', 0) for det in image_results)
                cv2.putText(img, f"Hornets: {hornet_count}", (10, img.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Parasite 개수 표시
                if parasite_image_results:
                    p_count = parasite_image_results.get('count', 0)
                    cv2.putText(img, f"Parasites: {p_count}", (10, img.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 3) 윈도우에 표시
                window_name = f"Hive {hive_id} Inference"
                cv2.imshow(window_name, img)
                cv2.waitKey(1)

                # 4) late fusion으로 최종 클래스 결정
                final_label = late_fusion(audio_results, image_results)
                print(f"[Worker] Fused final label: {final_label}")

                # 5) 최종 레이블만 POST
                resp = requests.post(
                    'http://192.168.35.79:8080/sensing/threat',
                    json={
                        'id': hive_id,
                        'label': final_label,
                        'hornet_count': hornet_count,
                        'parasite_count': parasite_image_results.get('count', 0),
                        'measuredAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                    },
                    timeout=10.0
                )
                print(f"[Worker] POST responded {resp.status_code}")

        except Exception as e:
            print(f"[Worker] Error: {e}")
        finally:
            inference_queue.task_done()


# 주기적 메시지 발행 스레드
def scheduled_publisher(client):
    topic = 'pi/request'
    while True:
        message = b'Scheduled request'
        client.publish(topic, message)
        print(f"[Publish] {topic}")
        time.sleep(5)

# 메인 함수
def main():
    client = mqtt.Client()
    client.on_message = on_message

    client.connect('localhost', 1883, 60)
    client.subscribe('pi/response', qos=1)

    threading.Thread(target=inference_worker, daemon=True).start()
    threading.Thread(target=scheduled_publisher, args=(client,), daemon=True).start()

    client.loop_forever()

if __name__ == '__main__':
    main()
