import cv2 as cv


cap = cv.VideoCapture(0)

face = cv.CascadeClassifier('face.xml')


def blur(img):
    (h, w) = img.shape[:2]
    dw = int(w / 3.0)
    dh = int(h / 3.0)

    if dw % 2 == 0:
        dw -= 1
    if dh % 2 == 0:
        dh -= 1

    return cv.GaussianBlur(img, (dw, dh), 0)


while True:
    ret, img = cap.read()

    faces = face.detectMultiScale(
        img, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        img[y:y+h, x:x+w] = blur(img[y:y+h, x:x+w])

    cv.imshow('cam', img)

    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv.destroyAllWindows
