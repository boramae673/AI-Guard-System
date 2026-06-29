from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import cv2
import time
import math
import os
import numpy as np
import logging

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)

# ==========================
# 모델 로드
# ==========================
model = YOLO("yolov8n.pt")

# 얼굴인식 가벼운 모델
face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=-1, det_size=(320, 320))

# ==========================
# 카메라
# ==========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ==========================
# 얼굴 등록
# ==========================
KNOWN_FACE_DIR = "known_faces"
known_faces = []


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_known_faces():
    global known_faces

    print("등록 얼굴 불러오는 중...")

    if not os.path.exists(KNOWN_FACE_DIR):
        print("known_faces 폴더가 없습니다.")
        return

    for person_name in os.listdir(KNOWN_FACE_DIR):
        person_path = os.path.join(KNOWN_FACE_DIR, person_name)

        if not os.path.isdir(person_path):
            continue

        for file_name in os.listdir(person_path):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_path, file_name)
            img = cv2.imread(img_path)

            if img is None:
                print("이미지 읽기 실패:", img_path)
                continue

            faces = face_app.get(img)

            if len(faces) == 0:
                print("얼굴 못 찾음:", img_path)
                continue

            face = faces[0]

            known_faces.append({
                "name": person_name,
                "embedding": face.embedding
            })

            print("등록 완료:", person_name, file_name)

    print("총 등록 얼굴 수:", len(known_faces))


def recognize_face(face_embedding):
    best_name = "Unknown"
    best_score = -1

    for known in known_faces:
        score = cosine_similarity(face_embedding, known["embedding"])

        if score > best_score:
            best_score = score
            best_name = known["name"]

    if best_score >= 0.45:
        return best_name, best_score

    return "Unknown", best_score


load_known_faces()

# ==========================
# 추적 변수
# ==========================
stay_times = {}

track_to_person = {}
person_last_pos = {}
person_last_seen = {}
person_names = {}

next_person_id = 1

MAX_DISTANCE = 80
MAX_MISSING_TIME = 30

dashboard_data = {
    "people": 0,
    "unknown": 0,
    "restricted": 0,
    "high": 0,
    "risk": "안전",
    "score": 0
}

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def find_face_in_person_box(faces, x1, y1, x2, y2):
    for face in faces:
        fx1, fy1, fx2, fy2 = map(int, face.bbox)

        face_cx = (fx1 + fx2) // 2
        face_cy = (fy1 + fy2) // 2

        if x1 < face_cx < x2 and y1 < face_cy < y2:
            return face

    return None


def generate_frames():
    global next_person_id, dashboard_data

    frame_count = 0
    last_faces = []

    while True:
        success, frame = cap.read()

        if not success:
            print("카메라 프레임을 가져오지 못했습니다.")
            break

        h, w = frame.shape[:2]

        # ==========================
        # 제한구역
        # ==========================
        zone_w = w // 4
        zone_h = int(h * 0.8)

        zone_x1 = w - zone_w - 20
        zone_y1 = 20
        zone_x2 = w - 20
        zone_y2 = zone_y1 + zone_h

        # ==========================
        # YOLO 추적
        # ==========================
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.5,
            imgsz=640,
            verbose=False
        )

        # ==========================
        # 얼굴인식 속도 개선
        # 30프레임마다 작은 화면으로 얼굴인식
        # ==========================
        frame_count += 1

        if frame_count % 30 == 0:
            small_frame = cv2.resize(frame, (320, 240))
            temp_faces = face_app.get(small_frame)

            for face in temp_faces:
                face.bbox *= 2

            last_faces = temp_faces

        faces = last_faces

        now = time.time()

        people_count = 0
        unknown_count = 0
        restricted_count = 0
        high_risk_count = 0

        max_score = 0
        overall_risk = "안전"

        used_person_ids = set()

        # 제한구역 표시
        overlay = frame.copy()
        cv2.rectangle(overlay, (zone_x1, zone_y1), (zone_x2, zone_y2), RED, -1)
        frame = cv2.addWeighted(overlay, 0.16, frame, 0.84, 0)
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), RED, 2)

        cv2.putText(
            frame,
            "RESTRICTED AREA",
            (zone_x1, zone_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            RED,
            2
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.int().cpu().tolist()
            people_count = len(boxes)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_pos = (center_x, center_y)

                # ==========================
                # PERSON ID 부여
                # ==========================
                if track_id in track_to_person:
                    person_id = track_to_person[track_id]

                    if person_id in used_person_ids:
                        person_id = next_person_id
                        next_person_id += 1
                        track_to_person[track_id] = person_id

                else:
                    best_id = None
                    best_dist = 999999

                    for old_id, old_pos in person_last_pos.items():
                        if old_id in used_person_ids:
                            continue

                        last_seen = person_last_seen.get(old_id, 0)

                        if now - last_seen <= MAX_MISSING_TIME:
                            d = distance(current_pos, old_pos)

                            if d < best_dist:
                                best_dist = d
                                best_id = old_id

                    if best_id is not None and best_dist < MAX_DISTANCE:
                        person_id = best_id
                    else:
                        person_id = next_person_id
                        next_person_id += 1

                    track_to_person[track_id] = person_id

                used_person_ids.add(person_id)

                person_last_pos[person_id] = current_pos
                person_last_seen[person_id] = now

                # ==========================
                # 얼굴 인증
                # ==========================
                face = find_face_in_person_box(faces, x1, y1, x2, y2)

                if face is not None:
                    name, sim = recognize_face(face.embedding)

                    if name != "Unknown":
                        person_names[person_id] = name

                person_name = person_names.get(person_id, "Unknown")
                is_authorized = person_name != "Unknown"

                # ==========================
                # 제한구역 진입
                # ==========================
                inside = (
                    zone_x1 < center_x < zone_x2
                    and
                    zone_y1 < center_y < zone_y2
                )

                stay_sec = 0

                if inside:
                    restricted_count += 1

                    if person_id not in stay_times:
                        stay_times[person_id] = now

                    stay_sec = int(now - stay_times[person_id])

                else:
                    if person_id in stay_times:
                        del stay_times[person_id]

                # ==========================
                # 위험도 분류
                # ==========================
                if inside and stay_sec >= 20:
                    box_color = RED
                    risk_text = "DANGER"
                    reason_text = "OVER 20 SEC"
                    score = 100
                    high_risk_count += 1
                    overall_risk = "위험"

                elif inside:
                    box_color = ORANGE
                    risk_text = "WARNING"
                    reason_text = "RESTRICTED AREA"
                    score = 60

                    if overall_risk != "위험":
                        overall_risk = "경고"

                elif not is_authorized:
                    box_color = YELLOW
                    risk_text = "CAUTION"
                    reason_text = "UNAUTHORIZED"
                    score = 30
                    unknown_count += 1

                    if overall_risk == "안전":
                        overall_risk = "주의"

                else:
                    box_color = GREEN
                    risk_text = "SAFE"
                    reason_text = "AUTHORIZED"
                    score = 0

                max_score = max(max_score, score)

                # ==========================
                # 카메라 화면 표시
                # ==========================
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                if person_name == "Unknown":
                    label_name = f"Person_{person_id:02d}"
                else:
                    label_name = person_name.upper()

                cv2.putText(
                    frame,
                    label_name,
                    (x1, y1 - 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    risk_text,
                    (x1, y1 - 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    reason_text,
                    (x1, y1 - 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2
                )

                if inside:
                    cv2.putText(
                        frame,
                        f"Stay: {stay_sec}s",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2
                    )

        dashboard_data = {
            "people": people_count,
            "unknown": unknown_count,
            "restricted": restricted_count,
            "high": high_risk_count,
            "risk": overall_risk,
            "score": max_score
        }

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/data")
def data():
    return jsonify(dashboard_data)


if __name__ == "__main__":
    print("웹 서버 시작")
    print("브라우저 주소: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )