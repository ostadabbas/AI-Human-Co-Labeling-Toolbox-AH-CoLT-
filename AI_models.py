import matplotlib.pyplot as plt
import numpy as np
import glob
import shutil
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
    predictions['all_names'] = frames_name

    print('Saving predictions to pickle archive')
    helpers.savepkl(predictions, resource, model)
    
    # We create a new directory ("-FAN") containing text files
    # with coordinates of the predicted landmarks, as well 
    # image files with the predicted landmarks painted onto
    # the faces. This allows for manual inspection or external
    # manipulation of the predictions.
    if os.path.exists(resource + '-FAN'):
        shutil.rmtree(resource + '-FAN')
    os.makedirs(resource + '-FAN')
    for i, image_file in enumerate(predictions['all_names']):
        predicted_keypoints = predictions['all_keyps'][1][i][0]
        print('Saving predictions in image and text formats for ' 
              + os.path.split(image_file)[1])
        image = plt.imread(image_file)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.plot(predicted_keypoints[0], predicted_keypoints[1],
                'ro', markersize = 2)
        base_path = os.path.join(resource + '-FAN',
                                 os.path.splitext(os.path.basename(image_file))[0])
        plt.savefig(base_path + '.jpg')
        plt.close()
        with open(base_path + '.txt', 'w') as output_text:
            output_text.write('file: ' + os.path.split(image_file)[1] + '\n')
            output_text.write('x: ' + str(list(predicted_keypoints[0])) + '\n')
            output_text.write('x: ' + str(list(predicted_keypoints[1])) + '\n')
