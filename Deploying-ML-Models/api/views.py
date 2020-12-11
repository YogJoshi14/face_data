### General imports ###
from django.shortcuts import render
from django.http import StreamingHttpResponse,HttpResponse
from werkzeug.utils import secure_filename
from rest_framework.response import Response
from rest_framework.decorators import api_view

from time import time
from time import sleep
import os
import json
import threading
import uuid


from imutils.video import VideoStream
import imutils
import numpy as np
import pandas as pd
import cv2

from imutils import face_utils
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import dlib

from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage

from google.cloud import storage

shape_x = 48
shape_y = 48
input_shape = (shape_x, shape_y, 1)
nClasses = 7

thresh = 0.25
frame_check = 20


global sess
global graph

sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
model = load_model(os.path.abspath('xception_2_58.h5'))

face_detect = dlib.get_frontal_face_detector()
predictor_landmarks = dlib.shape_predictor("src/models/face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def generate(vs):
    face_data = {}

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    output_path = os.path.abspath('api/static/output_{}'.format(video_name))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    processed_video = cv2.VideoWriter(output_path, codec, fps, (frame_width,frame_height))

    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = vs.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_180)

        count +=1
        # print(count)
        face_index = 0
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)
        # gray, detected_faces, coord = detect_face(frame)

        for (i, rect) in enumerate(rects):

            shape = predictor_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Identify face coordinates
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y + h, x:x + w]

            # Zoom on extracted face
            try:
                face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))
            except:
                continue
            # Cast type float
            face = face.astype(np.float32)

            # Scale
            face /= float(face.max())
            face = np.reshape(face.flatten(), (1, 48, 48, 1))

            # Make Prediction
            # with graph.as_default():
            #     prediction = model.predict(face)

            # with graph.as_default():
            #     set_session(sess)
            prediction = model.predict(face)

            prediction_result = np.argmax(prediction)

            # Rectangle around the face
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0),
            #             2)

            # for (j, k) in shape:
            #     cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)

            # 1. Add prediction probabilities
            # cv2.putText(frame, "----------------", (40, 100 + 180 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 0)
            cv2.putText(frame, "Emotional report : Face #" + str(i + 1), (40, 120 + 180 * i), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, (255,255,255), 1)
            cv2.putText(frame, "Angry : " + str(round(prediction[0][0], 3)), (40, 140 + 180 * i),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0,0,255), 1)
            cv2.putText(frame, "Disgust : " + str(round(prediction[0][1], 3)), (40, 160 + 180 * i),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (77,77,255), 1)
            cv2.putText(frame, "Fear : " + str(round(prediction[0][2], 3)), (40, 180 + 180 * i),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (153,51,0), 1)
            cv2.putText(frame, "Happy : " + str(round(prediction[0][3], 3)), (40, 200 + 180 * i),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0,204,0), 1)
            cv2.putText(frame, "Sad : " + str(round(prediction[0][4], 3)), (40, 220 + 180 * i),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (102,153,153), 1)
            cv2.putText(frame, "Surprise : " + str(round(prediction[0][5], 3)), (40, 240 + 180 * i),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0,255,255), 1)
            cv2.putText(frame, "Neutral : " + str(round(prediction[0][6], 3)), (40, 260 + 180 * i),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (230,238,255), 1)

            # 2. Annotate main image with a label
            # if prediction_result == 0:
            #     cv2.putText(frame, "Angry", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # elif prediction_result == 1:
            #     cv2.putText(frame, "Disgust", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # elif prediction_result == 2:
            #     cv2.putText(frame, "Fear", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # elif prediction_result == 3:
            #     cv2.putText(frame, "Happy", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # elif prediction_result == 4:
            #     cv2.putText(frame, "Sad", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # elif prediction_result == 5:
            #     cv2.putText(frame, "Surprise", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # else:
            #     cv2.putText(frame, "Neutral", (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            dislike = prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][4]
            like = prediction[0][3] + prediction[0][5] + prediction[0][6]
            
            if dislike > like:
                cv2.putText(frame, "Not Satisfactory", (shape[8][0] - 60,shape[8][1] + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,204, 204), 1)
            
            elif like > dislike:
                cv2.putText(frame, "Satisfactory", (shape[8][0] - 60,shape[8][1] + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,204, 204), 1)

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
            cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)

            # 4. Detect Nose
            nose = shape[nStart:nEnd]
            noseHull = cv2.convexHull(nose)
            cv2.drawContours(frame, [noseHull], -1, (255, 255, 255), 1)

            # 5. Detect Mouth
            mouth = shape[mStart:mEnd]
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (255, 255, 255), 1)

            # 6. Detect Jaw
            jaw = shape[jStart:jEnd]
            jawHull = cv2.convexHull(jaw)
            cv2.drawContours(frame, [jawHull], -1, (255, 255, 255), 1)

            # 7. Detect Eyebrows
            ebr = shape[ebrStart:ebrEnd]
            ebrHull = cv2.convexHull(ebr)
            cv2.drawContours(frame, [ebrHull], -1, (255, 255, 255), 1)
            ebl = shape[eblStart:eblEnd]
            eblHull = cv2.convexHull(ebl)
            cv2.drawContours(frame, [eblHull], -1, (255, 255, 255), 1)


##########Drawing a face pattern ###########

            cv2.line(frame, tuple(shape[0]), tuple(shape[26]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[0]), tuple(shape[17]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[16]), tuple(shape[26]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[21]), tuple(shape[22]), (255,255,255), 1)

            cv2.line(frame, tuple(shape[1]), tuple(shape[28]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[2]), tuple(shape[29]), (255,255,255), 1)

            cv2.line(frame, tuple(shape[15]), tuple(shape[28]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[14]), tuple(shape[29]), (255,255,255), 1)

            cv2.line(frame, tuple(shape[0]), tuple(shape[30]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[16]), tuple(shape[30]), (255,255,255), 1)

            cv2.line(frame, tuple(shape[31]), tuple(shape[52]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[35]), tuple(shape[50]), (255,255,255), 1)

            cv2.line(frame, tuple(shape[54]), tuple(shape[29]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[48]), tuple(shape[29]), (255,255,255), 1)

            cv2.line(frame, tuple(shape[30]), tuple(shape[31]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[30]), tuple(shape[35]), (255,255,255), 1)

            cv2.line(frame, tuple(shape[1]), tuple(shape[41]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[2]), tuple(shape[41]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[3]), tuple(shape[31]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[4]), tuple(shape[31]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[5]), tuple(shape[48]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[5]), tuple(shape[67]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[6]), tuple(shape[67]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[6]), tuple(shape[66]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[7]), tuple(shape[66]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[7]), tuple(shape[65]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[8]), tuple(shape[65]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[8]), tuple(shape[56]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[9]), tuple(shape[56]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[9]), tuple(shape[55]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[10]), tuple(shape[55]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[10]), tuple(shape[54]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[11]), tuple(shape[54]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[12]), tuple(shape[35]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[13]), tuple(shape[35]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[14]), tuple(shape[35]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[15]), tuple(shape[24]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[16]), tuple(shape[24]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[48]), tuple(shape[31]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[31]), tuple(shape[49]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[49]), tuple(shape[32]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[32]), tuple(shape[50]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[50]), tuple(shape[33]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[33]), tuple(shape[51]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[51]), tuple(shape[34]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[34]), tuple(shape[52]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[52]), tuple(shape[35]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[35]), tuple(shape[53]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[0]), tuple(shape[36]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[16]), tuple(shape[45]), (255,255,255), 1)
            
            cv2.line(frame, tuple(shape[17]), tuple(shape[36]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[36]), tuple(shape[18]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[18]), tuple(shape[37]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[37]), tuple(shape[19]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[19]), tuple(shape[38]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[38]), tuple(shape[20]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[20]), tuple(shape[39]), (255,255,255), 1)

            cv2.line(frame, tuple(shape[22]), tuple(shape[42]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[42]), tuple(shape[23]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[23]), tuple(shape[43]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[43]), tuple(shape[24]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[24]), tuple(shape[44]), (255,255,255), 1)
            cv2.line(frame, tuple(shape[44]), tuple(shape[25]), (255,255,255), 1)




        # cv2.putText(frame, 'Number of Faces : ' + str(len(rects)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)

        processed_video.write(frame)

        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", frame)

    processed_video.release()
    return output_path


@api_view(('GET','POST'))
def index(request):
    global video_name
    if request.method=="POST":
        if request.FILES['video']:
            video_file = request.FILES['video']
            video_ext = (os.path.splitext(video_file.name))[-1]
            print(video_ext)
            video_name = str(uuid.uuid4()) + str(video_ext)
            with open(os.path.abspath('api/static/{}'.format(video_name)), 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)

            vs = cv2.VideoCapture(os.path.abspath('api/static/{}'.format(video_name)))

            output_path = generate(vs)
            storage_client = storage.Client.from_service_account_json(os.path.abspath('My First Project-f16fe106e0bb.json'))
            bucket = storage_client.get_bucket("emotion_recognition_data")
            FILENAME = "frames/output_{}".format(video_name)
            blob = bucket.blob(FILENAME)
            blob.upload_from_filename(output_path)

            ##Problem of trailing "/" in cloud path

            cloud_path = "https://storage.googleapis.com/emotion_recognition_data/{}".format(FILENAME)

            os.remove(output_path)
            os.remove(os.path.abspath('api/static/{}'.format(video_name)))

            return HttpResponse(cloud_path)
