__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"
__email__ = "nguyenquangbinh803@gmail.com"
__copyright__ = "Copyright 2020"
__status__ = "Working on demo stage 2, develop the entire local server for all raspberry"
__version__ = "1.0.1"

import os
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import tensorflow as tf

import numpy as np
import SmartCamera_ShareMemory as sc_share_memory
import pickle

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import face_recognition
import glob
import time


class FaceDetectAndRecognition:

    def __init__(self):
        print("Init models")
        prototxtPath = "face_detector/deploy.prototxt"
        weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

        self.embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

        self.faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
        self.config = tf.ConfigProto(
            device_count={'GPU': 1},
            intra_op_parallelism_threads=1,
            allow_soft_placement=True
        )

        #
        # self.session = tf.Session(config=self.config)
        #
        # keras.backend.set_session(self.session)
        #
        # self.maskNet = tf.lite.Interpreter(model_path="model.tflite")
        # self.maskNet.allocate_tensors()
        #
        # self.input_details = self.maskNet.get_input_details()
        # self.output_details = self.maskNet.get_output_details()
        # self.input_shape = self.input_details[0]['shape']

    def detect_face(self, frame, rgb_require=True):

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        faces = []
        locs = []
        preds = []
        rgb_faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                sc_share_memory.human_appear_status = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                try:
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    rgb_faces.append(face)
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    faces.append(face)
                    sc_share_memory.face_area = (endX - startX) * (endY - startY)
                    if (endX - startX) * (endY - startY) > 3000:
                        sc_share_memory.face_detect_status = True
                    else:
                        sc_share_memory.face_detect_status = False

                    locs.append((startX, startY, endX, endY))
                except Exception as exp:
                    print(str(exp))
        if not locs:
            sc_share_memory.face_detect_status = False
            sc_share_memory.human_appear_status = False

        if sc_share_memory.face_detect_status:
            sc_share_memory.global_locs = locs
        if rgb_require:
            return locs, rgb_faces
        else:
            return locs, faces

    def encoding_with_torch_openface(self, dataset_directory):
        folders = glob.glob(os.path.abspath("") + "/" + dataset_directory + "/*/")
        knownNames = []
        knownEmbeddings = []

        for folder in folders:
            target_name = os.path.basename(os.path.normpath(folder))
            images = glob.glob(dataset_directory + "/" + target_name + "/*jpg")
            # print(folder)
            total = 0
            for image in images:

                img = cv2.imread(image)
                start = time.time()
                faceBlob = cv2.dnn.blobFromImage(img, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                face_detect.embedder.setInput(faceBlob)
                vec = face_detect.embedder.forward()
                print(target_name, time.time() - start)
                knownNames.append(target_name)
                knownEmbeddings.append(vec.flatten())
                total += 1

        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open("embeddings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()

    def encoding_with_dlib_face(self, dataset_directory):
        folders = glob.glob(os.path.abspath("") + "/" + dataset_directory + "/*/")
        knownNames = []
        knownEncodings = []

        for folder in folders:
            target_name = os.path.basename(os.path.normpath(folder))
            images = glob.glob(dataset_directory + "/" + target_name + "/*jpg")
            print(folder)
            total = 0
            for image in images:
                img = cv2.imread(image)
                locs, faces = self.detect_face(img)
                for index, face in enumerate(faces):
                    start = time.time()
                    print(locs[index][1], locs[index][2], locs[index][3], locs[index][0])
                    boxes = [(locs[index][1], locs[index][2], locs[index][3], locs[index][0])]
                    encodings = face_recognition.face_encodings(face, boxes)
                    for encoding in encodings:
                        knownEncodings.append(encoding)
                        knownNames.append(target_name)
                    print(time.time() - start)
                    total += 1

        data = {"encodings": knownEncodings, "names": knownNames}
        f = open("encodings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()

    def face_recognize_with_dlib(self, frame, encoding_data):
        locs, faces = self.detect_face(frame)
        encodings = []
        names = []
        boxes = []
        for index, face in enumerate(faces):
            print(locs[index][1], locs[index][2], locs[index][3], locs[index][0])
            boxes = [(locs[index][1], locs[index][2], locs[index][3], locs[index][0])]
            encodings = face_recognition.face_encodings(face, boxes)

        for encoding in encodings:
            print(encoding_data["encodings"], encoding)
            matches = face_recognition.compare_faces(encoding_data["encodings"],
                                                     encoding)
            name = "Unknown"

            if True in matches:

                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = encoding_data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            names.append(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

#    def recording_face(self, dataset_directory, id_directory, frame):



if __name__ == "__main__":

    face_detect = FaceDetectAndRecognition()
    # face_detect.encoding_with_dlib_face("dataset")
    cap = cv2.VideoCapture(-1)
    encoding_data = pickle.loads(open("encodings.pickle", "rb").read())
    id_directory = uuid.uuid1().hex
    dataset_directory = "dataset/"
    os.makedirs("dataset/" + id_directory)
    count = 0

    while True:
        ret, frame = cap.read()

        if ret:
            locs, faces = face_detect.detect_face(frame)
            for index, face in enumerate(faces):
                count += 1
                write_face = frame[locs[index][1]:locs[index][3], locs[index][0]:locs[index][2]]
                cv2.imwrite(dataset_directory + id_directory + "/" + str(count) + ".jpg", write_face)
                cv2.imshow("Recording", face)
                cv2.waitKey(1)
    # face_detect.face_recognize_with_dlib(frame, encoding_data)
