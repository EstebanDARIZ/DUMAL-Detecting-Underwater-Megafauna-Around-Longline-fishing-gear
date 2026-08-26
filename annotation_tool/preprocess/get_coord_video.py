import cv2


video_path = "./video_raw/Raie_event_2.0.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

if not ret:
    print("Erreur : impossible de lire la vidéo")
    exit()

image = frame.copy()
cv2.imwrite("./output/frame_000.jpg", image)

def mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img_copy = image.copy()
        text = f"X: {x}, Y: {y}"
        cv2.putText(img_copy, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Frame", img_copy)


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_move)
cv2.imshow("Frame", image)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()