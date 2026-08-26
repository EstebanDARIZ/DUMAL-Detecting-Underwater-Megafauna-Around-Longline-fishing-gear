import cv2


img = cv2.imread("./output/frame_000.jpg")
img_h, img_w, _ = img.shape

with open("./output/test_10/labels/frame_000000.txt", "r", encoding='utf-8') as f:
    line = f.readline()
    print(line)
    cls, xc, yc, w, h, scr = map(float, line.strip().split())    
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    print(x1, y1, x2, y2)

cv2.rectangle(img, (int(x1), int(y1)),(int(x2), int(y2)), (255, 0, 0), 5)
cv2.imshow("boxe", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


