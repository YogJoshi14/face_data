import tensorflow as tf
import cv2
from tensorflow.keras import datasets, layers, models,callbacks,preprocessing
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import time
import pickle
in_encoder = Normalizer()
out_encoder = LabelEncoder()

exp_model_path = 'C:/Users/Admin/PycharmProjects/face-recognition-opencv/models/expression_model.h5'
face_model_path = 'C:/Users/Admin/PycharmProjects/face-recognition-opencv/models/svc_model.sav'

e_model = models.load_model(exp_model_path)
f_model = pickle.load(open(face_model_path, 'rb'))
print(e_model.summary())
objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# def facecrop(imagepath):
#     facedata = "/content/drive/My Drive/Camera_detection/haarcascade_frontalface_alt.xml"
#     cascade = cv2.CascadeClassifier(facedata)
#     print(cascade.empty())
#     img = cv2.imread(imagepath)
#     print(img.shape)
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(gray.shape)
#
#     faces = cascade.detectMultiScale(gray)
#     for f in faces:
#         x, y, w, h = [v for v in f]
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     sub_face = gray[y:y + h, x:x + w]
#     img_resized = cv2.resize(sub_face, (48, 48))
#     x = preprocessing.image.img_to_array(img_resized)
#     x = np.expand_dims(x, axis=0)
#     print(x.shape)
#     # x= np.array(x, dtype=np.float)
#     # a = a.flatten()
#     x /= 255
#     custom = model.predict(x)
#     # print ("Writing: " + custom[0])
#     emotion_analysis(custom[0])
#     plt.imshow(img)
#     plt.show()
#
#
def emotion_analysis(emotions):
    y_pos = np.arange(len(objects))
    maxium = max(emotions)
    indx = emotions.index(maxium)
    return objects[indx]
    # plt.bar(y_pos, emotions, align='center', alpha=0.5)
    # plt.xticks(y_pos, objects)
    # plt.ylabel('percentage')
    # plt.title('emotion')
    # plt.show()
#
# # vs = cv2.VideoCapture(0)
# # time.sleep(1.0)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(ret)
while cap.isOpened() and ret == True:
    ret, frame = cap.read()
    # Load an image to entity from file
    img = frame.copy()
    facedata = "C:/Users/Admin/PycharmProjects/face-recognition-opencv/models/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)
    print(cascade.empty())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray)
    x= y= w= h = None
    # try:
    for f in faces:
        x, y, w, h = [v for v in f]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    sub_face = gray[y:y + h, x:x + w]
    # img_resized = cv2.resize(sub_face, (48, 48))
    # x = preprocessing.image.img_to_array(img_resized)
    # x = np.expand_dims(x, axis=0)
    sub_face_norm = in_encoder.transform(sub_face)
    samples = np.expand_dims(sub_face_norm, axis=0)
    yhat_class = f_model.predict(samples)
    # get name
    print(yhat_class)
    class_index = yhat_class[0]
    print('class:',class_index)
    predict_names = out_encoder.inverse_transform(yhat_class)
    print(predict_names)
    # all_names = out_encoder.inverse_transform([0, 1, 2, 3, 4])
    # print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    # print('Predicted: \n%s \n%s' % (all_names, yhat_prob[0] * 100))
    # print('Expected: %s' % random_face_name[0])
    # x= np.array(x, dtype=np.float)
    # a = a.flatten()
    # x /= 255
    # custom = e_model.predict(x)
    # # print(custom[1])
    # prob = custom[0].max()
    # clas = objects[custom[0].argmax()]
    # print(prob,clas)
    # label = emotion_analysis(custom[0])
    # cv2.putText(img, f'{str(clas)}', (int(50),int(50)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    # except:
    #     pass
    cv2.imshow('feed',img)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()