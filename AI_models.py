import numpy as np
import glob
import os

from Models.Hourglass.inference import hg_labeler
import helpers


def hourglass_model(resource, model):
    preds = hg_labeler(resource)

    # create structure for AI Labeler result
    frames_name = []
    frames_kpts = []
    im_list = sorted(glob.glob(os.path.join(resource, "*.jpg")))
    for i in range(len(preds)):
        sp_kpts = []
        frames_name.append(im_list[i])
        arr = np.empty([2, 16])
        for j in range(len(preds[0])):
            arr[0, j] = preds[i][j][0]
            arr[1, j] = preds[i][j][1]

        sp_kpts.append(arr)
        frames_kpts.append(sp_kpts)

    predictions = {}
    predictions['all_keyps'] = [[], frames_kpts]
    predictions['all_boxes'] = [[] for i in range(len(frames_kpts))]

    helpers.savepkl(predictions, resource, model)

def OpenCV_Model(resource, model):
    print("To do...")

def DetectAndTrack_Model(resource, model):
    print("To do...")