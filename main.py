from ultralytics import YOLO
import cv2
import time

# YOLO 모델 로드
model = YOLO("yolov8n.pt")

# 웹캠 연결
cap = cv2.VideoCapture(0)

# ID별 체류 시간 저장
stay_times = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    # ==========================
    # 제한구역 (오른쪽 세로형)
    # ==========================
    zone_w = w // 4          # 가로 폭 유지
    zone_h = int(h * 0.8)    # 화면 높이의 80%

    zone_x1 = w - zone_w - 20
    zone_y1 = 20

    zone_x2 = w - 20
    zone_y2 = zone_y1 + zone_h

    # 사람 추적
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    frame = results[0].plot()

    people_count = 0
    risk_score = 0

    # 제한구역 표시
    cv2.rectangle(
        frame,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "RESTRICTED AREA",
        (zone_x1, zone_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )

    if results[0].boxes is not None:

        people_count = len(results[0].boxes)

        ids = results[0].boxes.id

        if ids is not None:

            ids = ids.int().cpu().tolist()

            for box, track_id in zip(results[0].boxes, ids):

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 중심점 표시
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (255, 0, 0),
                    -1
                )

                # 제한구역 진입 여부
                inside = (
                    zone_x1 < center_x < zone_x2
                    and
                    zone_y1 < center_y < zone_y2
                )

                if inside:

                    risk_score += 30

                    # 처음 들어온 시간 저장
                    if track_id not in stay_times:
                        stay_times[track_id] = time.time()

                    stay_sec = int(
                        time.time() - stay_times[track_id]
                    )

                    # ID + 체류시간 표시
                    cv2.putText(
                        frame,
                        f"ID {track_id}: {stay_sec}s",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

                    # 5초 이상 체류
                    if stay_sec >= 5:

                        risk_score += 20

                        cv2.putText(
                            frame,
                            "WARNING!",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            3
                        )

                else:

                    # 구역 밖으로 나가면 시간 초기화
                    if track_id in stay_times:
                        del stay_times[track_id]

    # 사람 수 표시
    cv2.putText(
        frame,
        f"People: {people_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # 위험도 표시
    cv2.putText(
        frame,
        f"Risk Score: {risk_score}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # 화면 출력
    cv2.imshow(
        "AI CCTV Risk Analysis",
        frame
    )

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
