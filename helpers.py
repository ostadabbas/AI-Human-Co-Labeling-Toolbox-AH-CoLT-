import cv2
import numpy as np
import csv
import pickle

POSE_PAIRS = {"Mask R-CNN":[(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)],
              "Hourglass":[(0, 1), (1, 2), (2, 6), (3, 6), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (12, 7), (13, 7), (13, 14), (14, 15)]}
kpts_thres = 2


def visposes(lists_poses):
    pose_thres = 0.95
    vis_pose_idx = []
    for i in range(len(lists_poses)):
        if lists_poses[i, 4] > pose_thres:
            vis_pose_idx.append(i)
    return vis_pose_idx


def viskpts(img, lists_kpts, vis_pose_idx, model):
    kpt_thres = 2
    lists_vis_kpts = []
    flatten_vis_kpts = []
    if model == "Hourglass":
        vis_kpts = []
        for point in range(lists_kpts[0].shape[1]):
            if lists_kpts[0][0, point] >= 0 and lists_kpts[0][1, point] >= 0 and lists_kpts[0][0, point] <= img.shape[1]\
               and lists_kpts[0][1, point] <= img.shape[0]:
                vis_kpts.append(1)
                flatten_vis_kpts.append(1)
            else:
                vis_kpts.append(0)
                flatten_vis_kpts.append(0)
        lists_vis_kpts.append(vis_kpts)
    # elif len(lists_kpts) > 0 and model == "Mask R-CNN":
    #     for pose in range(len(lists_kpts)):
    #         vis_kpts = []
    #         for point in range(lists_kpts[0].shape[1]):
    #             if point == 3 or point == 4: # do not display ear kepoints
    #                 vis_kpts.append(0)
    #                 flatten_vis_kpts.append(0)
    #                 continue
    #             if lists_kpts[pose][2, point] > kpt_thres and pose in vis_pose_idx:
    #                 vis_kpts.append(1)
    #                 flatten_vis_kpts.append(1)
    #             else:
    #                 vis_kpts.append(0)
    #                 flatten_vis_kpts.append(0)
    #         lists_vis_kpts.append(vis_kpts)
    return lists_vis_kpts, flatten_vis_kpts

def drawkpts(img, lists_kpts, lists_vis, model):
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
                    cv2.circle(img_draw, (int(x_kpts), int(y_kpts)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(img_draw, str(num)+"_"+str(i), (int(points[0, i]), int(points[1, i])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2, bottomLeftOrigin = True)
                # plt.text(int(points[0, i]), int(points[1, i]), str(num)+"_"+str(i), color='b', fontsize=10)
                else:
                    points.append(None)
            pairs = POSE_PAIRS[model]
            # pairs = POSE_PAIRS["Mask R-CNN"]

            for pair in pairs:
                # print(pair)
                partA = pair[0]
                partB = pair[1]
                if points[partA] and points[partB]:
                    cv2.line(img_draw, points[partA], points[partB], (200, 200, 0), 2)

    return img_draw

def drawflags(img, points, flags, model, idx):
    h = img.shape[0]
    w = img.shape[1]
    img_draw = img.copy()
    for i in range(points.shape[1]):
        cv2.circle(img_draw, (int(points[0, i]), int(points[1, i])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(img_draw, str(idx), (int(points[0, i]), int(points[1, i])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 255, 255), 4, bottomLeftOrigin=True)

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


def csv2list(file):
    frames_name = []
    frames_kpts = []
    # np.empty([1,2])

    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            sp_kpts = []
            frames_name.append(row[0])
            arr = np.empty([2,16])
            arr[0,:] = row[1:17]
            arr[1,:] = row[17:35]

            # print(arr)
            sp_kpts.append(arr)
            frames_kpts.append(sp_kpts)
    f.close()
    return frames_name, frames_kpts


def list2csv(frames_kpts, bbox):
    arr = []
    for idx in range(len(frames_kpts)):
        num_pose = len(frames_kpts[idx])
        num_kpts = frames_kpts[idx][0].shape[1]
        list_x = []
        list_y = []
        list_visibility = []
        for pose in range(num_pose):
            for joint in range(num_kpts):
                list_x.append(frames_kpts[idx][pose][0][joint])
                list_y.append(frames_kpts[idx][pose][1][joint])
                if frames_kpts[idx][pose].shape[0] == 3:
                    list_visibility.append(frames_kpts[idx][pose][2][joint])
        list = [list_x, list_y, list_visibility, bbox[idx]]
        arr.append((list))
    return arr

def savepkl(data, resource, id):
    path = resource + "_" + id + ".pkl"
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def readpkl(file):
    # file = "./img_flag.pkl"
    with open(file, 'rb') as f:
        data = pickle.load(f)
    # frames_kpts = data['all_keyps'][1]
    # frames_boxes = data['all_bbox'][1]
    return data