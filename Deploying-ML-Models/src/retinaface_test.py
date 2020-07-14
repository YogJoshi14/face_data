import cv2

from retinaface import RetinaFace

detector = RetinaFace(quality="normal")

# same with cv2.imread,cv2.cvtColor
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(ret)
while cap.isOpened() and ret == True:
    ret, frame = cap.read()
    # Load an image to entity from file
    img = frame.copy()
# rgb_image = detector.read(img)

    faces = detector.predict(img)
# faces is list of face dictionary
# each face dictionary contains x1 y1 x2 y2 left_eye right_eye nose left_lip right_lip
# faces=[{"x1":20,"y1":32, ... }, ...]

    result_img = detector.draw(img, faces)

# save
# cv2.imwrite("result_img.jpg",result_img)

# show using cv2
    cv2.imshow("result",result_img)
    cv2.waitKey()