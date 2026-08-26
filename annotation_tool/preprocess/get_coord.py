import cv2


image = cv2.imread("./dataset/images/R_frame_13621.jpg")

def mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img_copy = image.copy()
        text = f"X: {x}, Y: {y}"
        
        cv2.putText(img_copy, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        cv2.imshow("Image", img_copy)


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_move)
cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()