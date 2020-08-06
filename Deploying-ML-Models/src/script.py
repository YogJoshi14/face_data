
import tensorflow as tf
import dlib
import cv2
import argparse
import face_recognition
from tensorflow.keras import datasets, layers, models,callbacks,preprocessing
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage
import matplotlib.pyplot as plt
import time
import pickle
from imutils import face_utils
import os
import h5py
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to input video")
ap.add_argument("-i", "--input", type=str,
	help="path to output video")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-p","--pickle",type=str)
ap.add_argument("-l","--landmark",type=str)
args = vars(ap.parse_args())
## in_encoder = Normalizer()
# out_encoder = LabelEncoder()
shape_x = 48
shape_y = 48
input_shape = (shape_x, shape_y, 1)
nClasses = 7

thresh = 0.25
frame_check = 20
data_faces = pickle.loads(open(args["pickle"], "rb").read())
e_model = models.load_model(args["model"])
# f_model = pickle.load(open(face_model_path, 'rb'))
# print(e_model.summary())
objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

(eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

face_detect = dlib.get_frontal_face_detector()
predictor_landmarks = dlib.shape_predictor(args['landmark'])
#faceCascade = cv2.CascadeClassifier(args['landmark'])
def rec_face(rgb):
    boxes = boxes = face_recognition.face_locations(rgb,model='cnn')
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data_faces["encodings"],encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data_faces["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
    return name

def detect_face(frame):

    # Cascade classifier pre-trained model
    # BGR -> Gray conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cascade MultiScale classifier
    detected_faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                                  minSize=(shape_x, shape_y),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
    coord = []

    for x, y, w, h in detected_faces:
        if w > 100:
            sub_img = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            coord.append([x, y, w, h])

    return gray, detected_faces, coord

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
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def detect_face(frame):

    # Cascade classifier pre-trained model
    cascPath = 'models/face_landmarks.dat'
    faceCascade = cv2.CascadeClassifier(cascPath)

    # BGR -> Gray conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cascade MultiScale classifier
    detected_faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                                  minSize=(shape_x, shape_y),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
    coord = []

    for x, y, w, h in detected_faces:
        if w > 100:
            sub_img = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            coord.append([x, y, w, h])

    return gray, detected_faces, coord
def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
    gray = faces[0]
    detected_face = faces[1]

    new_face = []

    for det in detected_face:
        # Region dans laquelle la face est détectée
        x, y, w, h = det
        # X et y correspondent à la conversion en gris par gray, et w, h correspondent à la hauteur/largeur

        # Offset coefficient, np.floor takes the lowest integer (delete border of the image)
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray transforme l'image
        extracted_face = gray[y + vertical_offset:y + h, x + horizontal_offset:x - horizontal_offset + w]

        # Zoom sur la face extraite
        new_extracted_face = zoom(extracted_face,
                                  (shape_x / extracted_face.shape[0], shape_y / extracted_face.shape[1]))
        # cast type float
        new_extracted_face = new_extracted_face.astype(np.float32)
        # scale
        new_extracted_face /= float(new_extracted_face.max())
        # print(new_extracted_face)

        new_face.append(new_extracted_face)

    return new_face


def On_camera_detection():
# vs = cv2.VideoCapture(0)
# time.sleep(1.0)
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


def video_detection(in_path,out_path):

    stream = cv2.VideoCapture(in_path)
    writer = None

        # loop over frames from the video file stream
    while True:
        #grab the next frame
        (grabbed, frame) = stream.read()
        # if the frame was not grabbed, then we have reached the
        # end of the stream
        if not grabbed:
            break
        rgb = cv2.resize(frame, (960,600))
        #boxes = face_recognition.face_locations(rgb,
        #                                        model='cnn')
        #gray,face,boxes = detect_face(rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)
        name = rec_face(rgb)
        draw_and_detect(rgb,gray,rects,name)

        if writer is None and out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(out_path, fourcc, 24,
                                     (rgb.shape[1], rgb.shape[0]), True)

        # if the writer is not None, write the frame with recognized
        # faces t odisk
        if writer is not None:
            writer.write(rgb)
    # close the video file pointers
    stream.release()

    # check to see if the video writer point needs to be released
    if writer is not None:
        writer.release()
def draw_and_detect(frame,gray,rects,name):
    for (i, rect) in enumerate(rects):

        shape = predictor_landmarks(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Identify face coordinates
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = gray[y:y + h, x:x + w]

        # Zoom on extracted face
        face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))

        # Cast type float
        face = face.astype(np.float32)

        # Scale
        face /= float(face.max())
        face = np.reshape(face.flatten(), (1, 48, 48, 1))

        # Make Prediction
        prediction = e_model.predict(face)
        prediction_result = np.argmax(prediction)
        label = objects[prediction_result]
        # Rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, "Face {}".format(name), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    2)

        for (j, k) in shape:
            cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)

        # 1. Add prediction probabilities
        cv2.putText(frame, "----------------", (40, 100 + 180 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        cv2.putText(frame, "Emotional report : Face #" + str(i + 1), (40, 120 + 180 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, 155, 0)
        cv2.putText(frame, "Angry : " + str(round(prediction[0][0], 3)), (40, 140 + 180 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        cv2.putText(frame, "Disgust : " + str(round(prediction[0][1], 3)), (40, 160 + 180 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        cv2.putText(frame, "Fear : " + str(round(prediction[0][2], 3)), (40, 180 + 180 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(frame, "Happy : " + str(round(prediction[0][3], 3)), (40, 200 + 180 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(frame, "Sad : " + str(round(prediction[0][4], 3)), (40, 220 + 180 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(frame, "Surprise : " + str(round(prediction[0][5], 3)), (40, 240 + 180 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(frame, "Neutral : " + str(round(prediction[0][6], 3)), (40, 260 + 180 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)

        # 2. Annotate main image with a label

        cv2.putText(frame, label, (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 3. Eye Detection and Blink Count
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Compute Eye Aspect Ratio
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # And plot its contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 4. Detect Nose
        nose = shape[nStart:nEnd]
        noseHull = cv2.convexHull(nose)
        cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

        # 5. Detect Mouth
        mouth = shape[mStart:mEnd]
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # 6. Detect Jaw
        jaw = shape[jStart:jEnd]
        jawHull = cv2.convexHull(jaw)
        cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)

        # 7. Detect Eyebrows
        ebr = shape[ebrStart:ebrEnd]
        ebrHull = cv2.convexHull(ebr)
        cv2.drawContours(frame, [ebrHull], -1, (0, 255, 0), 1)
        ebl = shape[eblStart:eblEnd]
        eblHull = cv2.convexHull(ebl)
        cv2.drawContours(frame, [eblHull], -1, (0, 255, 0), 1)
video_detection(args['input'],args['output'])
