from __future__ import print_function, absolute_import
import json
import torch.utils.data as data
import cv2

from Models.Hourglass.pose.utils.transforms import *
from Models.Detection.person_detection import detect_person


class Imgs(data.Dataset):
    def __init__(self, source, inp_res, out_res):
        self.img_folder = source # root image folders
        self.jsonfile = './Models/Hourglass/data/mpii/mpii_annotations.json'
        self.inp_res = inp_res
        self.out_res = out_res
        self.file_list = []

        for f in os.listdir(self.img_folder):
            if f.endswith('.jpg') or f.endswith('.png'):
                self.file_list.append(f)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_folder, self.file_list[index])

        # For single-person pose estimation with a centered/scaled figure
        img = load_image(img_path)  # CxHxW
        img = im_to_numpy(img)
        print(img.shape)

        c = []  # rough human position in the image
        s = 0.0  # person scale w.r.t. 200 px height

        # if image in MPII dataset, fetch the center and scale parameters from annotation file.
        # Otherwise, apply person detector to generate bounding box (TO DO)
        isMPII = False
        # with open(self.jsonfile) as anno_file:
        #     self.anno = json.load(anno_file)
        # for idx, val in enumerate(self.anno):
        #     if val['img_paths'] == self.file_list[index]:
        #         isMPII = True
        #         c = val['objpos']
        #         s = val['scale_provided']
        #         print("c", c)
        #         print("s", s)
        #         break
        if isMPII == False:
            print("Generate center and scale parameters by YOLO Detector")
            # detect person bounding box by yolo detector
            print(img_path)
            bbox = detect_person(img_path)
            if bbox == []:
                print('Do not detect person!')
                bbox = [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(img.shape[1]), torch.tensor(img.shape[0])]
            c, s = box2cs(bbox, img.shape[1], img.shape[0])
            # print("c",c)
            # print('s',s)

        r = 0
        c = torch.Tensor(c)
        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # Prepare image
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)

        # Meta info
        meta = {'index': index, 'center': c, 'scale': s}

        return inp, meta

    def __len__(self):
        return len(self.file_list)


def imgs(source, inp_res, out_res):
    return Imgs(source, inp_res, out_res)


imgs.njoints = 16  # ugly but works
