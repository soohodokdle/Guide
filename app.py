from flask import Flask, render_template, request, jsonify
import os
import cv2
import base64
import time
from openai import OpenAI

app = Flask(__name__)

# 이미지 인코딩을 담당하는 클래스
class ImageLoader:
    @staticmethod
    def encode_image(image_path):
        # 이미지 파일을 Base64로 인코딩하는 메서드
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# 비디오에서 프레임을 캡처하고 분석하는 클래스
class FrameCapturer:
    def __init__(self, video_url, save_dir='captured_frames'):
        self.save_dir = save_dir  # 캡처한 프레임을 저장할 디렉토리
        self.video_url = video_url  # 비디오 URL
        self.frame_count = 0  # 캡처한 프레임 수
        self.start_time = time.time()  # 시작 시간 기록
        self.capture = cv2.VideoCapture(video_url)  # 비디오 캡처 객체 생성

        # 저장 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def capture_frames(self, interval=10):
        while True:
            ret, frame = self.capture.read()  # 비디오에서 프레임 읽기
            if not ret:
                break  # 더 이상 읽을 프레임이 없으면 종료

            cv2.imshow("DroidCam", frame)  # 프레임을 화면에 표시

            current_time = time.time()  # 현재 시간 기록
            elapsed_time = current_time - self.start_time  # 경과 시간 계산

            if elapsed_time >= interval:
                # 지정한 간격이 경과했을 때 프레임 저장
                frame_filename = os.path.join(self.save_dir, f'frame_{self.frame_count:04d}.jpg')
                cv2.imwrite(frame_filename, frame)
                print(f"Saved {frame_filename}")
                
                # 저장한 프레임을 Base64로 인코딩
                encoded_image = ImageLoader.encode_image(frame_filename)
                gpt_api = GptAPI()
                # 인코딩된 이미지를 GPT-API로 분석
                result = gpt_api.analyze_image(encoded_image)
                print(f"Analysis Result: {result}")

                self.frame_count += 1  # 프레임 수 증가
                self.start_time = current_time  # 시작 시간 갱신

            if cv2.waitKey(1) == 27:
                break  # 'Esc' 키를 누르면 루프 종료

        self.cleanup()  # 정리 작업 수행

    def cleanup(self):
        if self.capture.isOpened():
            self.capture.release()  # 비디오 캡처 객체 해제
        cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

# GPT-API를 통해 이미지를 분석하는 클래스
class GptAPI:
    def __init__(self):
        self.client = OpenAI(api_key="")  # OpenAI 클라이언트 초기화 # Api 키 넣기.
        self.model = "gpt-4o"  # 사용할 모델 지정
        self.prompt = "해당 이미지를 시각장애인의 시야라고 가정할게. 해당 이미지에서 전봇대나 자전거, 사람 등 시각장애인에게 위협이 될 요소를 기억하고 거리를 확인해줘. 그리고 5미터 이내의 요소들만 '우측 3미터에 전봇대가 존재, 전방 좌측 5미터에 쓰레기통 존재.' 와 같은 형식으로 한 문장으로 말해줘. 그리고 시각장애인에게 위협이 될만한 것이 없으면 \'없음\'이라고 말해줘. "

    def analyze_image(self, encoded_image):
        # 이미지 분석을 위한 메시지 생성
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }]
        # OpenAI API 호출하여 이미지 분석
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )

        result = ''
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                string = chunk.choices[0].delta.content
                result = ''.join([result, string])
        return result  # 분석 결과 반환

# ------------------------------- 여기까지 18일까지 작성했던 코드고.

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    video_url = request.form['video_url']
    frame_capturer = FrameCapturer(video_url)
    frame_capturer.capture_frames()
    return jsonify({"status": "Capture started"})

if __name__ == "__main__":
    app.run(debug=True)
