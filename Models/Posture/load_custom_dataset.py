import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import random
import json
from .utils import resize_image, resize_rgbd_image
import glob
import pickle

class load_data(data.Dataset):
    def __init__(self, img_folder, anno_file, transforms = None):
        self.anno = {}
        with open(anno_file) as json_file:
            anno = json.load(json_file)
        for i in range(len(anno['images'])):
            if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                self.anno[anno['images'][i]['file_name']] = anno['images'][i]['posture']

        imgs = os.listdir(img_folder)
        self.imgs = []
        for j in range(len(imgs)):
            filename = imgs[j]
            if filename in list(self.anno.keys()):
                self.imgs.append(os.path.join(img_folder, imgs[j]))
        random.shuffle(self.imgs)
        print(len(self.imgs))
        self.transforms = transforms


    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_name = img_path.split('/')[-1]
        posture = self.anno[img_name]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        data = cv2.imread(img_path)
        if self.transforms:
            img = resize_image(data, 224)
            data = self.transforms(img)

        '''
        cv2.imshow(img_name, cv2.imread(os.path.join('./images/validate_infant_gt', img_name)))
        cv2.imshow('skel_'+ posture, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return data, label

    def __len__(self):
        return len(self.imgs)

class load_raw_data(data.Dataset):
    def __init__(self, img_folder, anno_file, transforms = None):
        self.anno = {}
        with open(anno_file) as json_file:
            anno = json.load(json_file)
        for i in range(len(anno['images'])):
            if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                self.anno[anno['images'][i]['file_name']] = anno['images'][i]['posture']

        imgs = os.listdir(img_folder)
        self.imgs = []
        for j in range(len(imgs)):
            filename = imgs[j]
            if filename in list(self.anno.keys()):
                self.imgs.append(os.path.join(img_folder, imgs[j]))
        random.shuffle(self.imgs)
        print(len(self.imgs))
        self.transforms = transforms


    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_name = img_path.split('/')[-1]
        posture = self.anno[img_name]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        data = cv2.imread(img_path)
        if self.transforms:
            img = resize_image(data, 224)
            data = self.transforms(img)

        '''
        cv2.imshow(img_name, cv2.imread(os.path.join('./images/validate_infant_gt', img_name)))
        cv2.imshow('skel_'+ posture, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return data, label

    def __len__(self):
        return len(self.imgs)


class load_rgbd_data(data.Dataset):
    def __init__(self, img_folder, anno_file, transforms = None):
        self.anno = {}
        with open(anno_file) as json_file:
            anno = json.load(json_file)
        for i in range(len(anno['images'])):
            if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                self.anno[anno['images'][i]['file_name']] = anno['images'][i]['posture']

        imgs = glob.glob(img_folder + '/*.npy')
        self.imgs = []
        for j in range(len(imgs)):
            filename = imgs[j].split('/')[-1][:-4] + '.jpg'
            if filename in list(self.anno.keys()):
                self.imgs.append(imgs[j])
        random.shuffle(self.imgs)
        print(len(self.imgs))
        self.transforms = transforms


    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_name = img_path.split('/')[-1][:-4] + '.jpg'
        posture = self.anno[img_name]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        data = np.load(img_path)
        if self.transforms:
            img = resize_rgbd_image(data, 224)
            data = self.transforms(img)

        '''
        cv2.imshow(img_name, cv2.imread(os.path.join('./images/validate_infant_gt', img_name)))
        cv2.imshow('skel_'+ posture, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return data, label

    def __len__(self):
        return len(self.imgs)


class load_kpts(data.Dataset):
    def __init__(self, pred_file, anno_file, transforms = None):
        self.anno = {}
        self.imgs = {}
        self.index = []
        with open(anno_file) as json_file:
            anno = json.load(json_file)
        for i in range(len(anno['images'])):
            if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                self.index.append(str(anno['images'][i]['id']))
                self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']

        self.kpts = {}
        with open(pred_file) as json_file:
            pred = json.load(json_file)
        for i in range(len(pred)):
            if str(pred[i]['image_id']) in self.index:
                self.tmp = pred[i]['keypoints']
                # # print(len(anno['annotations'][i]['keypoints']))
                self.tmp_list = []
                for j in range(len(pred[i]['keypoints'])//3):
                    if j > 4:
                        if self.tmp[j * 3 + 2] > 0.0:
                            self.tmp_list.append(self.tmp[j * 3])
                            self.tmp_list.append(self.tmp[j * 3 + 1])
                        else:
                            self.tmp_list.append(0)
                            self.tmp_list.append(0)
                # print(self.tmp_list)
                len_x = np.max(self.tmp_list[0::2]) - np.min(self.tmp_list[0::2])
                len_y = np.max(self.tmp_list[1::2]) - np.min(self.tmp_list[1::2])
                self.tmp_list[0::2] = [tmp_x/len_x - 0.5 for tmp_x in self.tmp_list[0::2]]
                self.tmp_list[1::2] = [tmp_y/len_y - 0.5 for tmp_y in self.tmp_list[1::2]]
                self.kpts[str(pred[i]['image_id'])] = self.tmp_list

        # random.shuffle(self.index)


    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        return data, label, img_name

    def __len__(self):
        return len(self.index)

class load_kpts_5class(data.Dataset):
    def __init__(self, pred_file, anno_file, transforms = None):
        self.anno = {}
        self.imgs = {}
        self.index = []
        with open(anno_file) as json_file:
            anno = json.load(json_file)
        for i in range(len(anno['images'])):
            if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                self.index.append(str(anno['images'][i]['id']))
                self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']

        self.kpts = {}
        with open(pred_file) as json_file:
            pred = json.load(json_file)
        for i in range(len(pred)):
            if str(pred[i]['image_id']) in self.index:
                self.tmp = pred[i]['keypoints']
                # # print(len(anno['annotations'][i]['keypoints']))
                self.tmp_list = []
                for j in range(len(pred[i]['keypoints'])//3):
                    if j > 4:
                        if self.tmp[j * 3 + 2] > 0.0:
                            self.tmp_list.append(self.tmp[j * 3])
                            self.tmp_list.append(self.tmp[j * 3 + 1])
                        else:
                            self.tmp_list.append(0)
                            self.tmp_list.append(0)
                # print(self.tmp_list)
                len_x = np.max(self.tmp_list[0::2]) - np.min(self.tmp_list[0::2])
                len_y = np.max(self.tmp_list[1::2]) - np.min(self.tmp_list[1::2])
                self.tmp_list[0::2] = [tmp_x/len_x - 0.5 for tmp_x in self.tmp_list[0::2]]
                self.tmp_list[1::2] = [tmp_y/len_y - 0.5 for tmp_y in self.tmp_list[1::2]]
                self.kpts[str(pred[i]['image_id'])] = self.tmp_list

        # random.shuffle(self.index)


    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        elif posture == 'All Fours':
            label = 4
        else:
            label = 5

        return data, label, img_name

    def __len__(self):
        return len(self.index)

class load_kpts_5class_test(data.Dataset):
    def __init__(self, pred_file, transforms = None):

        '''
        self.anno = {}
        self.imgs = {}
        self.index = []
        with open(anno_file) as json_file:
            anno = json.load(json_file)
        for i in range(len(anno['images'])):
            if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                self.index.append(str(anno['images'][i]['id']))
                self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
        '''
        self.index = []
        self.kpts = {}
        with open(pred_file) as json_file:
            pred = json.load(json_file)
        for i in range(len(pred)):
            self.index.append(str(pred[i]['image_id']))
            self.tmp = pred[i]['keypoints']
            # # print(len(anno['annotations'][i]['keypoints']))
            self.tmp_list = []
            for j in range(len(pred[i]['keypoints'])//3):
                if j > 4:
                    if self.tmp[j * 3 + 2] > 0.0:
                        self.tmp_list.append(self.tmp[j * 3])
                        self.tmp_list.append(self.tmp[j * 3 + 1])
                    else:
                        self.tmp_list.append(0)
                        self.tmp_list.append(0)
            # print(self.tmp_list)
            len_x = np.max(self.tmp_list[0::2]) - np.min(self.tmp_list[0::2])
            len_y = np.max(self.tmp_list[1::2]) - np.min(self.tmp_list[1::2])
            self.tmp_list[0::2] = [tmp_x/len_x - 0.5 for tmp_x in self.tmp_list[0::2]]
            self.tmp_list[1::2] = [tmp_y/len_y - 0.5 for tmp_y in self.tmp_list[1::2]]
            self.kpts[str(pred[i]['image_id'])] = self.tmp_list

        # random.shuffle(self.index)


    def __getitem__(self, index):
        idx = self.index[index]
        img_id = idx
        data = self.kpts[idx]
        data = torch.Tensor(data)

        return data, img_id

    def __len__(self):
        return len(self.index)

class load_pkl_kpts_test(data.Dataset):
    def __init__(self, pred_file, transforms = None):
        with open(pred_file, 'rb') as f:
            data = pickle.load(f)
        frames_kpts = data['all_keyps'][1]
        frames_names = data['all_frames']
        self.index = []
        self.kpts = {}
        self.imgs = {}
        for i in range(len(frames_kpts)):
            self.index.append(str(i))
            self.tmp = frames_kpts[i][0]
            self.imgs[str(i)] = frames_names[i]
            # # print(len(anno['annotations'][i]['keypoints']))
            self.tmp_list = []
            for j in range(17):
                if j > 4:
                    self.tmp_list.append(self.tmp[0, j])
                    self.tmp_list.append(self.tmp[1, j])
            # print(self.tmp_list)
            len_x = np.max(self.tmp_list[0::2]) - np.min(self.tmp_list[0::2])
            len_y = np.max(self.tmp_list[1::2]) - np.min(self.tmp_list[1::2])
            self.tmp_list[0::2] = [tmp_x/len_x - 0.5 for tmp_x in self.tmp_list[0::2]]
            self.tmp_list[1::2] = [tmp_y/len_y - 0.5 for tmp_y in self.tmp_list[1::2]]
            self.kpts[str(i)] = self.tmp_list

        # random.shuffle(self.index)


    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]

        return data, img_name

    def __len__(self):
        return len(self.index)

class load_kpts_test(data.Dataset):
    def __init__(self, pred_file, transforms = None):

        '''
        self.anno = {}
        self.imgs = {}
        self.index = []
        with open(anno_file) as json_file:
            anno = json.load(json_file)
        for i in range(len(anno['images'])):
            if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                self.index.append(str(anno['images'][i]['id']))
                self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
        '''
        self.index = []
        self.kpts = {}
        with open(pred_file) as json_file:
            pred = json.load(json_file)
        for i in range(len(pred)):
            self.index.append(str(pred[i]['image_id']))
            self.tmp = pred[i]['keypoints']
            # # print(len(anno['annotations'][i]['keypoints']))
            self.tmp_list = []
            for j in range(len(pred[i]['keypoints'])//3):
                if j > 4:
                    if self.tmp[j * 3 + 2] > 0.0:
                        self.tmp_list.append(self.tmp[j * 3])
                        self.tmp_list.append(self.tmp[j * 3 + 1])
                    else:
                        self.tmp_list.append(0)
                        self.tmp_list.append(0)
            # print(self.tmp_list)
            len_x = np.max(self.tmp_list[0::2]) - np.min(self.tmp_list[0::2])
            len_y = np.max(self.tmp_list[1::2]) - np.min(self.tmp_list[1::2])
            self.tmp_list[0::2] = [tmp_x/len_x - 0.5 for tmp_x in self.tmp_list[0::2]]
            self.tmp_list[1::2] = [tmp_y/len_y - 0.5 for tmp_y in self.tmp_list[1::2]]
            self.kpts[str(pred[i]['image_id'])] = self.tmp_list

        # random.shuffle(self.index)


    def __getitem__(self, index):
        idx = self.index[index]
        img_id = idx
        data = self.kpts[idx]
        data = torch.Tensor(data)

        return data, img_id

    def __len__(self):
        return len(self.index)


class load_kpts_depth(data.Dataset):
    def __init__(self, pred_file, anno_file, depth_dir, transforms = None):
        self.anno = {}
        self.depth = {}
        self.index = []
        with open(anno_file) as json_file:
            anno = json.load(json_file)

        for i in range(len(anno['images'])):
            if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                id = str(anno['images'][i]['id'])
                self.index.append(id)
                self.anno[id] = anno['images'][i]['posture']

                h = anno['images'][i]['height']
                w = anno['images'][i]['width']
                ori_img_name = anno['images'][i]['original_file_name']
                subject = ori_img_name[0:8]
                frame = ori_img_name[-9:-4]
                depth_filename = subject + '-' + frame + '.txt'
                depth_file = open(os.path.join(depth_dir, depth_filename), 'r')
                depth_list = depth_file.read().splitlines()
                depth_list = [elem for elem in depth_list if elem != '']
                depth_list = list(map(int, depth_list))
                depth_data = np.reshape(depth_list, (h, w))
                self.depth[id] = depth_data

        self.kpts = {}
        with open(pred_file) as json_file:
            pred = json.load(json_file)
        for i in range(len(pred)):
            if str(pred[i]['image_id']) in self.index:
                tmp = pred[i]['keypoints']
                # # print(len(anno['annotations'][i]['keypoints']))
                tmp_list = []
                for j in range(len(pred[i]['keypoints'])//3):
                    if j > 4:
                        if tmp[3 * j + 2] > 0.0:
                            x = tmp[3 * j]
                            y = tmp[3 * j + 1]
                            d = self.depth[str(pred[i]['image_id'])][int(y)][int(x)]
                            tmp_list.append(x)
                            tmp_list.append(y)
                            tmp_list.append(d)
                        else:
                            tmp_list.append(0)
                            tmp_list.append(0)
                            tmp_list.append(0)
                # print(self.tmp_list)
                len_x = np.max(tmp_list[0::3]) - np.min(tmp_list[0::3])
                len_y = np.max(tmp_list[1::3]) - np.min(tmp_list[1::3])
                len_d = np.max(tmp_list[2::3]) - np.min(tmp_list[1::3])
                tmp_list[0::3] = [tmp_x/len_x - 0.5 for tmp_x in tmp_list[0::3]]
                tmp_list[1::3] = [tmp_y /len_y - 0.5 for tmp_y in tmp_list[1::3]]
                tmp_list[2::3] = [tmp_d /len_d - 0.5 for tmp_d in tmp_list[2::3]]
                self.kpts[str(pred[i]['image_id'])] = tmp_list

        random.shuffle(self.index)
        print(len(self.index))


    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        return data, label

    def __len__(self):
        return len(self.index)


class load_kpts_gt(data.Dataset):
    def __init__(self, anno_file, transforms = None):
        self.anno = {}
        self.imgs = {}
        self.index = []
        self.kpts = {}
        with open(anno_file) as json_file:
            anno = json.load(json_file)
        for i in range(len(anno['images'])):
            if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                self.index.append(str(anno['images'][i]['id']))
                self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
                self.tmp_list = []
                self.tmp = anno['annotations'][i]['keypoints']
                for j in range(len(self.tmp) // 3):
                    if j > 4:
                        if int(self.tmp[j * 3 + 2]) > 0.0:
                            self.tmp_list.append(self.tmp[j * 3])
                            self.tmp_list.append(self.tmp[j * 3 + 1])
                        else:
                            self.tmp_list.append(0)
                            self.tmp_list.append(0)
                # print(self.tmp_list)
                len_x = np.max(self.tmp_list[0::2]) - np.min(self.tmp_list[0::2])
                len_y = np.max(self.tmp_list[1::2]) - np.min(self.tmp_list[1::2])
                self.tmp_list[0::2] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::2]]
                self.tmp_list[1::2] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::2]]
                self.kpts[str(anno['images'][i]['id'])] = self.tmp_list


    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        return data, label, img_name

    def __len__(self):
        return len(self.index)

class load_kpts_5class_gt(data.Dataset):
    def __init__(self, anno_file, transforms = None):
        self.anno = {}
        self.imgs = {}
        self.index = []
        self.kpts = {}
        with open(anno_file) as json_file:
            anno = json.load(json_file)
        for i in range(len(anno['images'])):
            if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                self.index.append(str(anno['images'][i]['id']))
                self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
                self.tmp_list = []
                self.tmp = anno['annotations'][i]['keypoints']
                for j in range(len(self.tmp) // 3):
                    if j > 4:
                        if int(self.tmp[j * 3 + 2]) > 0.0:
                            self.tmp_list.append(self.tmp[j * 3])
                            self.tmp_list.append(self.tmp[j * 3 + 1])
                        else:
                            self.tmp_list.append(0)
                            self.tmp_list.append(0)
                # print(self.tmp_list)
                len_x = np.max(self.tmp_list[0::2]) - np.min(self.tmp_list[0::2])
                len_y = np.max(self.tmp_list[1::2]) - np.min(self.tmp_list[1::2])
                self.tmp_list[0::2] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::2]]
                self.tmp_list[1::2] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::2]]
                self.kpts[str(anno['images'][i]['id'])] = self.tmp_list


    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        elif posture == 'All Fours':
            label = 4
        else:
            label = 5

        return data, label, img_name

    def __len__(self):
        return len(self.index)

'''
class load_kpts3D(data.Dataset):
    def __init__(self, anno_file, kpt_file, train = True, transforms = None):
        self.anno = {}
        self.imgs = {}
        self.index = []
        self.kpts = {}
        if train:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            for i in range(len(anno['images'])):
                if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                    self.index.append(str(anno['images'][i]['id']))
                    self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                    self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
                    self.tmp_list = []
                    self.tmp = anno['annotations'][i]['3d_keypoints']
                    # print('cccccccccccc', len(self.tmp))
                    for j in range(len(self.tmp) // 4):
                        if j > 4:  # eyes and ears
                            if int(self.tmp[j * 4 + 3]) > 0.0:
                                self.tmp_list.append(self.tmp[j * 4]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 1]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 2]/1000)
                            else:
                                self.tmp_list.append(0)
                                self.tmp_list.append(0)
                                self.tmp_list.append(0)
                    # print(self.tmp_list)
                    len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                    len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                    len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                    self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                    self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                    self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                    self.kpts[str(anno['images'][i]['id'])] = self.tmp_list
                    # print('aaaaaaaaaaaaa', len(self.tmp_list))
        else:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            kpts3d = np.load(kpt_file)   #(215,14,3)
            print(np.shape(kpts3d))
            print(np.shape(anno['images']))

            for i in range(len(anno['images'])-1):
                if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                    print(anno['images'][i]['file_name'])
                    self.index.append(str(anno['images'][i]['id']))
                    self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                    self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
                    self.tmp_list = []
                    self.kpts_tmp = kpts3d[i]   # (14,3)
                    self.tmp = np.zeros((12,3))
                    # self.tmp[0,:] = (self.kpts_tmp[13,:] + self.kpts_tmp[12,:])/2
                    # self.tmp[1:5,:] = 0
                    self.tmp[0,:] = self.kpts_tmp[9,:]
                    self.tmp[1,:] = self.kpts_tmp[8,:]
                    self.tmp[2,:] = self.kpts_tmp[10,:]
                    self.tmp[3,:] = self.kpts_tmp[7,:]
                    self.tmp[4,:] = self.kpts_tmp[11,:]
                    self.tmp[5,:] = self.kpts_tmp[6,:]

                    self.tmp[6,:] = self.kpts_tmp[3,:]
                    self.tmp[7,:] = self.kpts_tmp[2,:]
                    self.tmp[8,:] = self.kpts_tmp[4,:]
                    self.tmp[9,:] = self.kpts_tmp[1,:]
                    self.tmp[10,:] = self.kpts_tmp[5,:]
                    self.tmp[11,:] = self.kpts_tmp[0,:]
         
     
                    self.tmp_list = [j for sub in self.tmp for j in sub]
                    # print(self.tmp_list)
                    len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                    len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                    len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                    self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                    self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                    self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                    self.kpts[str(anno['images'][i]['id'])] = self.tmp_list
                    # print('bbbbbbbbbbbbbbb', len(self.tmp_list))

    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        return data, label

    def __len__(self):
        return len(self.index)
'''

class load_kpts3D_syn(data.Dataset):
    def __init__(self, anno_file, ds_name, train = True, transforms = None):
        self.anno = {}
        self.imgs = {}
        self.index = []
        self.kpts = {}
        if train:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            for i in range(len(anno['images'])):
                self.index.append(str(anno['images'][i]['id']))
                self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
                self.tmp_list = []
                self.tmp = anno['annotations'][i]['3d_keypoints']
                # print('cccccccccccc', len(self.tmp))
                for j in range(len(self.tmp) // 4):
                    if j > 4:  # eyes and ears  
                        if int(self.tmp[j * 4 + 3]) > 0.0:
                            if ds_name == 'mimm':
                                self.tmp_list.append(self.tmp[j * 4]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 1]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 2]/1000)
                            else:
                                self.tmp_list.append(self.tmp[j * 4]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 1]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 2]/1000)
                        else:
                            self.tmp_list.append(0)
                            self.tmp_list.append(0)
                            self.tmp_list.append(0)
                # print(self.tmp_list)
                len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                self.kpts[str(anno['images'][i]['id'])] = self.tmp_list
                # print('aaaaaaaaaaaaa', len(self.tmp_list))
    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        return data, label, img_name

    def __len__(self):
        return len(self.index)
       
class load_kpts3D_syn_eval(data.Dataset):
    def __init__(self, anno_file, kpt_file, eval = True, transforms = None):
        self.anno = {}
        self.imgs = {}
        self.index = []
        self.kpts = {}
        if eval:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            kpts3d = np.load(kpt_file)   #(215,14,3)
            
            print(len(kpts3d[0]))
            for j in range((len(anno['images']))):
                self.index.append(str(j))
                self.anno[str(j)] = anno['images'][j]['posture']
                self.imgs[str(j)] = anno['images'][j]['file_name']
                self.tmp_list = []
                self.kpts_tmp = kpts3d[j]   # (14,3)
                self.tmp = np.zeros((12,3))

                self.tmp[0,:] = self.kpts_tmp[9,:]
                self.tmp[1,:] = self.kpts_tmp[8,:]
                self.tmp[2,:] = self.kpts_tmp[10,:]
                self.tmp[3,:] = self.kpts_tmp[7,:]
                self.tmp[4,:] = self.kpts_tmp[11,:]
                self.tmp[5,:] = self.kpts_tmp[6,:]

                self.tmp[6,:] = self.kpts_tmp[3,:]
                self.tmp[7,:] = self.kpts_tmp[2,:]
                self.tmp[8,:] = self.kpts_tmp[4,:]
                self.tmp[9,:] = self.kpts_tmp[1,:]
                self.tmp[10,:] = self.kpts_tmp[5,:]
                self.tmp[11,:] = self.kpts_tmp[0,:]


                self.tmp_list = [j for sub in self.tmp for j in sub]

                len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                self.kpts[str(j)] = self.tmp_list
    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        return data, label, img_name

    def __len__(self):
        return len(self.index)

class load_kpts3D(data.Dataset):
    def __init__(self, anno_file, kpt_file, name_file, train = True, transforms = None):
        self.anno = {}
        self.imgs = {}
        self.index = []
        self.kpts = {}
        if train:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            for i in range(len(anno['images'])):
                if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                    self.index.append(str(anno['images'][i]['id']))
                    self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                    self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
                    self.tmp_list = []
                    self.tmp = anno['annotations'][i]['3d_keypoints']
                    #print('cccccccccccc', len(self.tmp))
                    for j in range(len(self.tmp) // 4):
                        if j > 4:  # eyes and ears
                            if int(self.tmp[j * 4 + 3]) > 0.0:
                                self.tmp_list.append(self.tmp[j * 4]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 1]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 2]/1000)
                            else:
                                self.tmp_list.append(0)
                                self.tmp_list.append(0)
                                self.tmp_list.append(0)
                    # print(self.tmp_list)
                    len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                    len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                    len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                    self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                    self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                    self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                    self.kpts[str(anno['images'][i]['id'])] = self.tmp_list
                    # print('aaaaaaaaaaaaa', len(self.tmp_list))
        else:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            kpts3d = np.load(kpt_file)   #(215,14,3)
            name_list = np.load(name_file)
            print('name_list: ', np.shape(name_list))
            print('input kpts: ', np.shape(kpts3d))
            print(np.shape(anno['images']))

            for j in range(len(name_list)):
                for i in range((len(anno['images']))):
                    if name_list[j] == anno['images'][i]['file_name']: 
                        if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown' and anno['images'][i]['file_name'][0:8] != 'MIMM5027':
                            # print(anno['images'][i]['file_name'])
                            self.index.append(str(j))
                            self.anno[str(j)] = anno['images'][i]['posture']
                            self.imgs[str(j)] = anno['images'][i]['file_name']
                            self.tmp_list = []
                            self.kpts_tmp = kpts3d[j]   # (14,3)
                            self.tmp = np.zeros((12,3))

                            self.tmp[0,:] = self.kpts_tmp[9,:]
                            self.tmp[1,:] = self.kpts_tmp[8,:]
                            self.tmp[2,:] = self.kpts_tmp[10,:]
                            self.tmp[3,:] = self.kpts_tmp[7,:]
                            self.tmp[4,:] = self.kpts_tmp[11,:]
                            self.tmp[5,:] = self.kpts_tmp[6,:]

                            self.tmp[6,:] = self.kpts_tmp[3,:]
                            self.tmp[7,:] = self.kpts_tmp[2,:]
                            self.tmp[8,:] = self.kpts_tmp[4,:]
                            self.tmp[9,:] = self.kpts_tmp[1,:]
                            self.tmp[10,:] = self.kpts_tmp[5,:]
                            self.tmp[11,:] = self.kpts_tmp[0,:]


                            self.tmp_list = [j for sub in self.tmp for j in sub]

                            len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                            len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                            len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                            self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                            self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                            self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                            self.kpts[str(j)] = self.tmp_list
             
        print(len(self.index))

    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        return data, label, img_name

    def __len__(self):
        return len(self.index)


class load_kpts3D_syrip(data.Dataset):
    def __init__(self, anno_file, kpt_file, name_file, transforms = None):
        self.anno = {}
        self.imgs = {}
        self.index = []
        self.kpts = {}
        train = False
        if train:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            for i in range(len(anno['images'])):
                if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                    self.index.append(str(anno['images'][i]['id']))
                    self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                    self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
                    self.tmp_list = []
                    self.tmp = anno['annotations'][i]['3d_keypoints']
                    #print('cccccccccccc', len(self.tmp))
                    for j in range(len(self.tmp) // 4):
                        if j > 4:  # eyes and ears
                            if int(self.tmp[j * 4 + 3]) > 0.0:
                                self.tmp_list.append(self.tmp[j * 4]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 1]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 2]/1000)
                            else:
                                self.tmp_list.append(0)
                                self.tmp_list.append(0)
                                self.tmp_list.append(0)
                    # print(self.tmp_list)
                    len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                    len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                    len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                    self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                    self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                    self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                    self.kpts[str(anno['images'][i]['id'])] = self.tmp_list
                    # print('aaaaaaaaaaaaa', len(self.tmp_list))
        else:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            kpts3d = np.load(kpt_file)   #(215,14,3)
            name_list = np.load(name_file)
            print('name_list: ', np.shape(name_list))
            print('input kpts: ', np.shape(kpts3d))
            print(np.shape(anno['images']))

            for j in range(len(name_list)):
                for i in range((len(anno['images']))):
                    if name_list[j] == anno['images'][i]['file_name']: 
                        if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown' and anno['images'][i]['file_name'][0:8] != 'MIMM5027':
                            # print(anno['images'][i]['file_name'])
                            self.index.append(str(j))
                            self.anno[str(j)] = anno['images'][i]['posture']
                            self.imgs[str(j)] = anno['images'][i]['file_name']
                            self.tmp_list = []
                            self.kpts_tmp = kpts3d[j]   # (14,3)
                            self.tmp = np.zeros((12,3))

                            self.tmp[0,:] = self.kpts_tmp[9,:]
                            self.tmp[1,:] = self.kpts_tmp[8,:]
                            self.tmp[2,:] = self.kpts_tmp[10,:]
                            self.tmp[3,:] = self.kpts_tmp[7,:]
                            self.tmp[4,:] = self.kpts_tmp[11,:]
                            self.tmp[5,:] = self.kpts_tmp[6,:]

                            self.tmp[6,:] = self.kpts_tmp[3,:]
                            self.tmp[7,:] = self.kpts_tmp[2,:]
                            self.tmp[8,:] = self.kpts_tmp[4,:]
                            self.tmp[9,:] = self.kpts_tmp[1,:]
                            self.tmp[10,:] = self.kpts_tmp[5,:]
                            self.tmp[11,:] = self.kpts_tmp[0,:]


                            self.tmp_list = [j for sub in self.tmp for j in sub]

                            len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                            len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                            len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                            self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                            self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                            self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                            self.kpts[str(j)] = self.tmp_list
             
        print(len(self.index))

    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        else:
            label = 4

        return data, label, img_name

    def __len__(self):
        return len(self.index)



class load_kpts3D_syrip_5class(data.Dataset):
    def __init__(self, anno_file, kpt_file, name_file, transforms = None):
        self.anno = {}
        self.imgs = {}
        self.index = []
        self.kpts = {}
        train = False
        if train:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            for i in range(len(anno['images'])):
                if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                    self.index.append(str(anno['images'][i]['id']))
                    self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                    self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
                    self.tmp_list = []
                    self.tmp = anno['annotations'][i]['3d_keypoints']
                    #print('cccccccccccc', len(self.tmp))
                    for j in range(len(self.tmp) // 4):
                        if j > 4:  # eyes and ears
                            if int(self.tmp[j * 4 + 3]) > 0.0:
                                self.tmp_list.append(self.tmp[j * 4]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 1]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 2]/1000)
                            else:
                                self.tmp_list.append(0)
                                self.tmp_list.append(0)
                                self.tmp_list.append(0)
                    # print(self.tmp_list)
                    len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                    len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                    len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                    self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                    self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                    self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                    self.kpts[str(anno['images'][i]['id'])] = self.tmp_list
                    # print('aaaaaaaaaaaaa', len(self.tmp_list))
        else:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            kpts3d = np.load(kpt_file)   #(215,14,3)
            name_list = np.load(name_file)
            print('name_list: ', np.shape(name_list))
            print('input kpts: ', np.shape(kpts3d))
            print(np.shape(anno['images']))

            for j in range(len(name_list)):
                for i in range((len(anno['images']))):
                    if name_list[j] == anno['images'][i]['file_name']: 
                        if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown' and anno['images'][i]['file_name'][0:8] != 'MIMM5027':
                            # print(anno['images'][i]['file_name'])
                            self.index.append(str(j))
                            self.anno[str(j)] = anno['images'][i]['posture']
                            self.imgs[str(j)] = anno['images'][i]['file_name']
                            self.tmp_list = []
                            self.kpts_tmp = kpts3d[j]   # (14,3)
                            self.tmp = np.zeros((12,3))

                            self.tmp[0,:] = self.kpts_tmp[9,:]
                            self.tmp[1,:] = self.kpts_tmp[8,:]
                            self.tmp[2,:] = self.kpts_tmp[10,:]
                            self.tmp[3,:] = self.kpts_tmp[7,:]
                            self.tmp[4,:] = self.kpts_tmp[11,:]
                            self.tmp[5,:] = self.kpts_tmp[6,:]

                            self.tmp[6,:] = self.kpts_tmp[3,:]
                            self.tmp[7,:] = self.kpts_tmp[2,:]
                            self.tmp[8,:] = self.kpts_tmp[4,:]
                            self.tmp[9,:] = self.kpts_tmp[1,:]
                            self.tmp[10,:] = self.kpts_tmp[5,:]
                            self.tmp[11,:] = self.kpts_tmp[0,:]


                            self.tmp_list = [j for sub in self.tmp for j in sub]

                            len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                            len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                            len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                            self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                            self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                            self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                            self.kpts[str(j)] = self.tmp_list
             
        print(len(self.index))

    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
        posture = self.anno[idx]
        if posture == 'Supine':
            label = 0
        elif posture == 'Prone':
            label = 1
        elif posture == 'Sitting':
            label = 2
        elif posture == 'Standing':
            label = 3
        elif posture == 'All Fours':
            label = 4
        else:
            label = 5

        return data, label, img_name

    def __len__(self):
        return len(self.index)

class load_kpts3D_syrip_5class_test(data.Dataset):
    def __init__(self, kpt_file, name_file, transforms = None):
        self.imgs = {}
        self.index = []
        self.kpts = {}
        train = False
        if train:
            with open(anno_file) as json_file:
                anno = json.load(json_file)

            for i in range(len(anno['images'])):
                if anno['images'][i]['posture'] != 'None' and anno['images'][i]['posture'] != 'Unknown':
                    self.index.append(str(anno['images'][i]['id']))
                    self.anno[str(anno['images'][i]['id'])] = anno['images'][i]['posture']
                    self.imgs[str(anno['images'][i]['id'])] = anno['images'][i]['file_name']
                    self.tmp_list = []
                    self.tmp = anno['annotations'][i]['3d_keypoints']
                    #print('cccccccccccc', len(self.tmp))
                    for j in range(len(self.tmp) // 4):
                        if j > 4:  # eyes and ears
                            if int(self.tmp[j * 4 + 3]) > 0.0:
                                self.tmp_list.append(self.tmp[j * 4]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 1]/1000)
                                self.tmp_list.append(self.tmp[j * 4 + 2]/1000)
                            else:
                                self.tmp_list.append(0)
                                self.tmp_list.append(0)
                                self.tmp_list.append(0)
                    # print(self.tmp_list)
                    len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                    len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                    len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                    self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                    self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                    self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                    self.kpts[str(anno['images'][i]['id'])] = self.tmp_list
                    # print('aaaaaaaaaaaaa', len(self.tmp_list))
        else:
            kpts3d = np.load(kpt_file)   #(215,14,3)
            name_list = np.load(name_file)
            print('name_list: ', np.shape(name_list))
            print('input kpts: ', np.shape(kpts3d))

            for j in range(len(name_list)):
                self.imgs[str(j)] = name_list[j]
                self.index.append(str(j))
                self.tmp_list = []
                self.kpts_tmp = kpts3d[j]   # (14,3)
                self.tmp = np.zeros((12,3))

                self.tmp[0,:] = self.kpts_tmp[9,:]
                self.tmp[1,:] = self.kpts_tmp[8,:]
                self.tmp[2,:] = self.kpts_tmp[10,:]
                self.tmp[3,:] = self.kpts_tmp[7,:]
                self.tmp[4,:] = self.kpts_tmp[11,:]
                self.tmp[5,:] = self.kpts_tmp[6,:]

                self.tmp[6,:] = self.kpts_tmp[3,:]
                self.tmp[7,:] = self.kpts_tmp[2,:]
                self.tmp[8,:] = self.kpts_tmp[4,:]
                self.tmp[9,:] = self.kpts_tmp[1,:]
                self.tmp[10,:] = self.kpts_tmp[5,:]
                self.tmp[11,:] = self.kpts_tmp[0,:]


                self.tmp_list = [j for sub in self.tmp for j in sub]

                len_x = np.max(self.tmp_list[0::3]) - np.min(self.tmp_list[0::3])
                len_y = np.max(self.tmp_list[1::3]) - np.min(self.tmp_list[1::3])
                len_z = np.max(self.tmp_list[2::3]) - np.min(self.tmp_list[2::3])
                self.tmp_list[0::3] = [tmp_x / len_x - 0.5 for tmp_x in self.tmp_list[0::3]]
                self.tmp_list[1::3] = [tmp_y / len_y - 0.5 for tmp_y in self.tmp_list[1::3]]
                self.tmp_list[2::3] = [tmp_z / len_z - 0.5 for tmp_z in self.tmp_list[2::3]]
                self.kpts[str(j)] = self.tmp_list

    def __getitem__(self, index):
        idx = self.index[index]
        data = self.kpts[idx]
        data = torch.Tensor(data)
        img_name = self.imgs[idx]
 
        return data, img_name

    def __len__(self):
        return len(self.index)
    def __getitem__(self, index):
        idx = self.index[index]
        img_id = idx
        data = self.kpts[idx]
        data = torch.Tensor(data)

        return data, img_id

    def __len__(self):
        return len(self.index)
