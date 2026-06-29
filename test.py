from ultralytics import YOLO
import cv2
import time
import math

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("카메라 열림:", cap.isOpened())

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

stay_times = {}

track_to_person = {}
person_last_pos = {}
person_last_seen = {}

next_person_id = 1

MAX_DISTANCE = 80
MAX_MISSING_TIME = 30

# 색상 BGR
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)

# 현재는 얼굴 인증 기능이 없으므로 전부 미인증 처리
authorized_person_ids = set()


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라 프레임을 가져오지 못했습니다.")
        break

    h, w = frame.shape[:2]

    # 제한구역 설정
    zone_w = w // 4
    zone_h = int(h * 0.8)

    zone_x1 = w - zone_w - 20
    zone_y1 = 20
    zone_x2 = w - 20
    zone_y2 = zone_y1 + zone_h

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        conf=0.15,
        iou=0.4,
        imgsz=960,
        verbose=False
    )

    now = time.time()

    people_count = 0
    unknown_count = 0
    restricted_count = 0
    high_risk_count = 0

    overall_risk = "SAFE"
    overall_color = GREEN

    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), RED, 2)

    cv2.putText(
        frame,
        "RESTRICTED AREA",
        (zone_x1, zone_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        RED,
        2
    )

    used_person_ids = set()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist()
        people_count = len(boxes)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_pos = (center_x, center_y)

            # PERSON ID 부여
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

            # 인증 여부
            is_authorized = person_id in authorized_person_ids

            # 제한구역 진입 여부
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

            # 위험도 분류
            if inside and stay_sec >= 20:
                risk_level = "DANGER"
                risk_reason = "20s+ IN AREA"
                risk_score = 100
                box_color = RED
                high_risk_count += 1

            elif inside:
                risk_level = "WARNING"
                risk_reason = "RESTRICTED AREA"
                risk_score = 60
                box_color = ORANGE

            elif not is_authorized:
                risk_level = "CAUTION"
                risk_reason = "UNAUTHORIZED"
                risk_score = 30
                box_color = YELLOW
                unknown_count += 1

            else:
                risk_level = "SAFE"
                risk_reason = "AUTHORIZED"
                risk_score = 0
                box_color = GREEN

            # 전체 위험도 갱신
            if risk_score == 100:
                overall_risk = "DANGER"
                overall_color = RED
            elif risk_score == 60 and overall_risk != "DANGER":
                overall_risk = "WARNING"
                overall_color = ORANGE
            elif risk_score == 30 and overall_risk not in ["DANGER", "WARNING"]:
                overall_risk = "CAUTION"
                overall_color = YELLOW

            # 박스 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            cv2.circle(frame, current_pos, 4, BLUE, -1)

            cv2.putText(
                frame,
                f"PERSON ID {person_id}",
                (x1, y1 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            cv2.putText(
                frame,
                risk_level,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            cv2.putText(
                frame,
                risk_reason,
                (x1, y1 - 10),
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

    # 상단 정보 표시
    cv2.putText(frame, f"People: {people_count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    cv2.putText(frame, f"Unknown: {unknown_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

    cv2.putText(frame, f"Restricted: {restricted_count}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2)

    cv2.putText(frame, f"High Risk: {high_risk_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

    cv2.putText(frame, f"Overall Risk: {overall_risk}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, overall_color, 3)

    cv2.imshow("AI CCTV Risk Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()