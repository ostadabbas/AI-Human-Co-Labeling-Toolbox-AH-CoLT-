import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import helpers

POSE_PAIRS = {"Mask R-CNN":[(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)],
              "densepose":[]}

def VisPoses(lists_poses):
    vis_pose_idx = []
    for i in range(len(lists_poses)):
        if lists_poses[i, 4] > 0.95:
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
                if point == 3 or point == 4:
                    vis_kpts.append(0)
                    flatten_vis_kpts.append(0)
                    continue
                if lists_kpts[pose][2,point] > kpt_thres and pose in vis_pose_idx:
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

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (255, 0, 0)
    if num_poses > 0 :
        for num in range(num_poses):
            points = []
            for i in range(num_kpts):
                if lists_vis[num][i] == 1:
                    x_kpts = lists_kpts[num][0, i]
                    y_kpts = lists_kpts[num][1, i]
                    points.append((int(x_kpts), int(y_kpts)))
                    cv2.circle(img_draw, (int(x_kpts), int(y_kpts)), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    bottomLeftCornerOfText = (int(x_kpts), int(y_kpts))
                    cv2.putText(img_draw, str(num)+"_"+str(i), bottomLeftCornerOfText, font, fontScale, fontColor, thickness=1, lineType=cv2.FILLED)
                else:
                    points.append(None)
            # print(points)
            pairs = POSE_PAIRS[model]
            pairs = POSE_PAIRS["Mask R-CNN"]

            for pair in pairs:
                partA = pair[0]
                partB = pair[1]
                if points[partA] and points[partB]:
                    cv2.line(img_draw, points[partA], points[partB], (200, 200, 0), 2)

    return img_draw

def DisplayKpts(kpts_file, img_folder, dis_folder, model):
    with open(kpts_file, 'rb') as f:
        pred = pickle.load(f)

    # load images list
    im_list = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))

    frames_kpts = pred['all_keyps'][1]
    num_frames = len(frames_kpts)
    print("Total frames: ", num_frames)

    frames_boxes = pred['all_boxes'][1]

    idx = 0
    end_idx = num_frames
    while idx <  end_idx:
        flag = []
        txt_list = []
        num_kpts = 0
        num_poses = 0
        vis_idx = []
        vis_pose_idx = []

        # load current image
        im_name = os.path.basename(im_list[idx])
        img = cv2.imread(im_list[idx])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # load current boxes
        lists_poses = frames_boxes[idx]
        vis_pose_idx = VisPoses(lists_poses)

        # load current keypoints
        lists_kpts = frames_kpts[idx]
        lists_vis, flatten_vis = VisKpts(lists_kpts, vis_pose_idx)

        num_poses = len(lists_kpts)
        num_kpts = lists_kpts[0].shape[1]
        vis_idx = np.nonzero(flatten_vis)[0]

        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # fig.canvas.set_window_title(im_name)

        # draw keypoints on image
        img = DrawKpts(img, lists_kpts, lists_vis, model)
        idx = idx + 1

        # save image
        dis_file = os.path.join(dis_folder, im_name)
        cv2.imwrite(dis_file, img)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_folder = './MPII_images'
    model = "Mask R-CNN"

    AI_file = 'res/MPII_images_detections.pkl'
    # create AI prediction visualization folder
    AI_folder = os.path.join(img_folder, 'AI_Vis_1')
    if not os.path.exists(AI_folder):
        os.mkdir(AI_folder)
    DisplayKpts(AI_file, img_folder, AI_folder, model)

    Res_file = 'res/v1/MPII_images_gt.pkl'
    # create Revised Kpts visualization folder
    Res_folder = os.path.join(img_folder, 'Res_Vis_1')
    if not os.path.exists(Res_folder):
        os.mkdir(Res_folder)
    DisplayKpts(Res_file, img_folder, Res_folder, model)


