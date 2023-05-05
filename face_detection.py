import cv2

face_cascade = cv2.CascadeClassifier("./assets/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        detected = frame[y : y + h, x : x + w]
        detected = cv2.GaussianBlur(detected, (23, 23), 30)
        frame[y : y + h, x : x + w] = detected

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.imwrite("capture_image.jpg", frame)

cap.release()
cv2.destroyAllWindows()


# Press Q To Quit
