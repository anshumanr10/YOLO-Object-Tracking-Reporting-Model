import cv2

cap = cv2.VideoCapture("rtsp://172.26.144.1:8554/live")

if not cap.isOpened():
    print("Failed to open OBS RTSP stream")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("OBS RTSP Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
