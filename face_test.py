import face_recognition
import cv2

known_image = face_recognition.load_image_file("KakaoTalk_20260604_212812833.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for (top, right, bottom, left), face_encoding in zip(locations, encodings):

        match = face_recognition.compare_faces(
            [known_encoding],
            face_encoding
        )[0]

        name = "UNKNOWN"

        if match:
            name = "SEOYUN"

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()