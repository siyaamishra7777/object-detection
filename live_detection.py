import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

print("Live object detection started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    results = model.predict(frame, conf=0.35, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Live Object Detection - Press Q to Quit", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
