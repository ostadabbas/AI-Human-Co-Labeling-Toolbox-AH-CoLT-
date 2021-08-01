import numpy as np
import glob
import os

from Models.Hourglass.inference import hg_labeler
from Models.Detectron2.demo import inference
from Models.FAN.inference import fan_labeler
import helpers


def hourglass_model(resource, model):
    preds = hg_labeler(resource)

    # create structure for AI Labeler result
    frames_name = []
    frames_kpts = []
    types = ('*.jpg', '*.png', '*.jpeg')
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(os.path.join(resource, files)))
    im_list = sorted(files_grabbed)
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

def detectron2_model(resource, model):
    preds = inference(resource)
    # create structure for AI Labeler result
    frames_name = []
    frames_kpts = []
    types = ('*.jpg', '*.png', '*.jpeg')
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(os.path.join(resource, files)))
    im_list = sorted(files_grabbed)
    for i in range(len(preds)):
        sp_kpts = []
        frames_name.append(im_list[i])
        arr = np.empty([2, 17])
        filename = os.path.basename(im_list[i])
        print(filename)
        print(len(preds[filename]['keypoints']))	
        if len(preds[filename]['keypoints']) == 0:
            os.remove(os.path.join(resource, filename))
            continue        
            
     
        for j in range(len(preds[filename]['keypoints'][0])):          
            arr[0, j] = preds[filename]['keypoints'][0][j][0]
            arr[1, j] = preds[filename]['keypoints'][0][j][1]
        
        sp_kpts.append(arr)
        frames_kpts.append(sp_kpts)

    predictions = {}
    predictions['all_keyps'] = [[], frames_kpts]
    predictions['all_boxes'] = [[] for i in range(len(frames_kpts))]

    helpers.savepkl(predictions, resource, model)

def FAN_model(resource, model):                 
    preds = fan_labeler(resource)

    # create structure for AI Labeler result
    frames_name = []
    frames_kpts = []
    types = ('*.jpg', '*.png', '*.jpeg')
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(os.path.join(resource, files)))
    im_list = sorted(files_grabbed)
    for i in range(len(preds)):
        sp_kpts = []
        frames_name.append(im_list[i])
        arr = np.empty([2, 68])
        for j in range(len(preds[0])):
            arr[0, j] = preds[i][j][0]
            arr[1, j] = preds[i][j][1]

        sp_kpts.append(arr)
        frames_kpts.append(sp_kpts)

    predictions = {}
    predictions['all_keyps'] = [[], frames_kpts]
    predictions['all_boxes'] = [[] for i in range(len(frames_kpts))]

    helpers.savepkl(predictions, resource, model)
