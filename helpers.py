import cv2
import os
import numpy as np
import matplotlib as plt

POSE_PAIRS = {"Mask R-CNN":[(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)],
              "densepose":[]}
kpts_thres = 2
def init_global(self):
    # initialize global variables for each image
    global curr_filename, curr_img, AI_kpts, id_label, flag, fix, fix_points
    curr_filename = None
    curr_img = None
    AI_kpts = None
    id_label = []
    flag = []
    fix = []
    fix_points = []

def npy2dict(arr,item):
    dict = {}
    lists = []
    list = arr.tolist()

    lists.append(list)
    lists.append(list)
    dict[item] = lists
    return dict

def VisPoses(lists_poses):
    pose_thres = 0.95
    vis_pose_idx = []
    for i in range(len(lists_poses)):
        if lists_poses[i, 4] > pose_thres:
            vis_pose_idx.append(i)
    return vis_pose_idx

def VisKpts(lists_kpts, vis_pose_idx):
    kpt_thres = 2
    lists_vis_kpts = []
    flatten_vis_kpts = []
    if len(lists_kpts) > 0:
        for pose in range(len(lists_kpts)):
            vis_kpts = []
            for point in range(lists_kpts[0].shape[1]):
                if point == 3 or point == 4: # do not display ear kepoints
                    vis_kpts.append(0)
                    flatten_vis_kpts.append(0)
                    continue
                if lists_kpts[pose][2, point] > kpt_thres and pose in vis_pose_idx:
                    vis_kpts.append(1)
                    flatten_vis_kpts.append(1)
                else:
                    vis_kpts.append(0)
                    flatten_vis_kpts.append(0)
            lists_vis_kpts.append(vis_kpts)
    return lists_vis_kpts, flatten_vis_kpts

def DrawKpts(img, lists_kpts, lists_vis, model):
    h = img.shape[0]
    w = img.shape[1]
    img_draw = img.copy()
    num_poses = len(lists_kpts)
    num_kpts = lists_kpts[0].shape[1]
    if num_poses > 0 :
        for num in range(num_poses):
            points = []
            for i in range(num_kpts):
                if lists_vis[num][i] == 1:
                    x_kpts = lists_kpts[num][0, i]
                    y_kpts = lists_kpts[num][1, i]
                    points.append((int(x_kpts), int(y_kpts)))
                    # points = np.append([x_kpts], [y_kpts], axis=0)
                    cv2.circle(img_draw, (int(x_kpts), int(y_kpts)), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(img_draw, str(num)+"_"+str(i), (int(points[0, i]), int(points[1, i])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2, bottomLeftOrigin = True)
                # plt.text(int(points[0, i]), int(points[1, i]), str(num)+"_"+str(i), color='b', fontsize=10)
                else:
                    points.append(None)
            print(points)
            pairs = POSE_PAIRS[model]
            # pairs = POSE_PAIRS["Mask R-CNN"]

            for pair in pairs:
                print(pair)
                partA = pair[0]
                partB = pair[1]
                if points[partA] and points[partB]:
                    cv2.line(img_draw, points[partA], points[partB], (200, 200, 0), 2)

    return img_draw

def DrawFlags(img, points, flags, model, idx):
    h = img.shape[0]
    w = img.shape[1]
    img_draw = img.copy()
    for i in range(points.shape[1]):
        cv2.circle(img_draw, (int(points[0, i]), int(points[1, i])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(img_draw, str(idx), (int(points[0, i]), int(points[1, i])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 255, 255), 2, bottomLeftOrigin=True)

        pairs = POSE_PAIRS[model]

        for pair in pairs:
            print(pair)
            partA = pair[0]
            partB = pair[1]
            cv2.line(img_draw, (int(points[0, partA]), int(points[1, partA])),
                     (int(points[0, partB]), int(points[1, partB])), (200, 200, 0), 2)

        if flags[i] == 0:
            cv2.putText(img_draw, str(i)+"_Wrong", (int(points[0, i]), int(points[1, i])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
        else:
            cv2.putText(img_draw, str(i)+"_Right", (int(points[0, i]), int(points[1, i])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

    return img_draw

