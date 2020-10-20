import face_alignment
from skimage import io
import os
import numpy as np
import cv2

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,device='cpu', flip_input=False) # can change into gpu if cuda enabled

def fan_labeler(source):
    preds=[]
    for filename in os.listdir(source):
        input_img = cv2.cvtColor(cv2.imread(source + "/" + filename),cv2.COLOR_BGR2RGB)
        preds.append(fa.get_landmarks(input_img)[-1])
    preds=np.array(preds)
    return preds
