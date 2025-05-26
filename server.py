import asyncio
import json
import base64
import httpx
from hbmqtt.broker import Broker
from hbmqtt.client import MQTTClient
from predict import predict_audio_bytes

# Broker 설정
broker_config = {
    'listeners': {
        'default': {
            'type': 'tcp',
            'bind': '0.0.0.0:1883'
        }
    },
    'sys_interval': 10,
    'topic-check': {
        'enabled': False
    }
}

# 요청 대기 큐 생성
inference_queue: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue()

# 요청을 큐에 쌓기만 하는 핸들러
async def handle_request(topic: str, payload: bytes):
    print(f"[Enqueue] Topic: {topic}")
    await inference_queue.put((topic, payload))

# 큐에 쌓인 요청을 순차적으로 처리하는 워커
async def inference_worker():
    print("[Worker] Inference worker started.")
    while True:
        topic, payload = await inference_queue.get()
        try:
            print(f"[Process] Handling topic: {topic}")
            # 1) JSON 파싱
            data = json.loads(payload.decode('utf-8'))
            # 2) base64로 인코딩된 audio 스트림 디코딩
            audio_b64 = data.get('audio')
            if not audio_b64:
                print("[Worker] Error: 'audio' 키 없음")
                continue
            audio_bytes = base64.b64decode(audio_b64)
            # 3) AI 추론 실행
            results = predict_audio_bytes(
                audio_bytes=audio_bytes,
                model_path='bee_hornet_cnn.h5'
            )
            print(f"[Worker] Prediction done: {results}")
            # 4) 결과 웹 서버에 전송 via FastAPI
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'http://localhost:8080/detect/hornet',
                    json={'predictions': results},
                    timeout=10.0
                )
                print(f"[Worker] POST responded {response.status_code}")
        except Exception as e:
            print(f"[Worker] Error processing request: {e}")
        finally:
            inference_queue.task_done()

# 브로커 시작
async def start_broker():
    broker = Broker(broker_config)
    await broker.start()

# 1분마다 토픽으로 요청 발행
async def scheduled_publisher():
    client = MQTTClient()
    await client.connect('mqtt://localhost:1883/')
    topic = 'pi/request'
    try:
        while True:
            message = b"Scheduled request"
            await client.publish(topic, message, qos=1)
            print(f"[Publish] {topic}")
            await asyncio.sleep(60)
    finally:
        await client.disconnect()

# 외부 요청 수신 및 큐에 적재
async def subscriber():
    client = MQTTClient()
    await client.connect('mqtt://localhost:1883/')
    await client.subscribe([('pi/response', 1)])
    print("[Subscriber] Subscribed to pi/response")
    try:
        while True:
            message = await client.deliver_message()
            packet = message.publish_packet
            topic = packet.variable_header.topic_name
            payload = packet.payload.data
            await handle_request(topic, payload)
    except asyncio.CancelledError:
        pass
    finally:
        await client.disconnect()

# 메인 진입점
async def main():
    await asyncio.gather(
        start_broker(),
        scheduled_publisher(),
        subscriber(),
        inference_worker()
    )

if __name__ == '__main__':
    asyncio.run(main())