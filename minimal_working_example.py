''' All Credits goes to https://github.com/vjgpt/Face-and-Emotion-Recognition '''
# cv2 (opencv)
# numpy
# dlib
# face_recognition 
# keras
# statistics
# utils


import cv2
import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognition
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input


# TO SET CORRECTLY
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes 
# this is because you wanna look at "average" emotion within some time window (aka number of frames), i.e., the mode, 
# because the method is not 100% accurate for every single frame. 
# E.g., for a time window of 5 frames, if you get, "happy"x2 times, "sad"x1 time, and "happy"x2 times again, 
# then the mode is "happy", and "sad" was a fluke.
emotion_window = []


# frame can be a static image, or a frame from a video stream
def get_emotion_from_frame(frame): 

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# this is gonna detect faces
    faces = detector(rgb_image)

    print ('Detected', len(faces), 'face(s)')

    # for every set of face coordinates detected, 
    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)

        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)


        emotion_probability = np.max(emotion_prediction)
        # now only the emotion predicted as most probable is returned, but you can 
        # consider computing a probability-weighted modes by taking in the emotion window
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
  
  		# debug: using x1 coordinate to understand who's left and who's right
        print (x1, emotion_text)

        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
            # now you may wanna log somewhere the emotion mode (perhaps linked to who's left and who's right)
        except:
            continue


# Test
# frame is now a static image, but could be a video frame 
frame = cv2.imread('./images/trump_obama.jpg')
get_emotion_from_frame(frame)
print(emotion_window)