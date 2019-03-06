#Script to run DetectandTrack Code on a single video which may or may not belong to any dataset.


import os
import os.path as osp
import sys

import numpy as np
import pickle
import cv2
import argparse
import shutil
import yaml
import glob
import time
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test_engine import initialize_model_from_cfg, empty_results, extend_results
from core.test import im_detect_all
from core.tracking_engine import _load_det_file, _write_det_file, _center_detections, _get_high_conf_boxes, _compute_matches

import utils.image as image_utils
import utils.video as video_utils
import utils.vis as vis_utils
import utils.subprocess as subprocess_utils
from utils.io import robust_pickle_dump
import utils.c2

try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass



def parse_args():
    parser = argparse.ArgumentParser(description='Run DetectandTrack on a single video and visualize the results')
    parser.add_argument(
        '--cfg', '-c', dest='cfg_file', required=True,
        help='Config file to run')
    parser.add_argument(
        '--img_fol', '-imf', dest='im_folder_path',
        help='Path to Image Folder',
        required=True)
    parser.add_argument(
        '--output', '-o', dest='out_path',
        help='Path to Output')
    parser.add_argument(
        '--visualize', '-vis', dest='visualize', default=False,
        help = 'set if you want to visualize keypoints on images')
    parser.add_argument(
        'opts', help='See lib/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def _id_or_index(ix, val):
    if len(val) == 0:
        return val
    else:
        return val[ix]


def _generate_visualizations(entry, ix, all_boxes, all_keyps, thresh = 0.9):
    im = cv2.imread(entry)
    cls_boxes_i = [
        _id_or_index(ix, all_boxes[j]) for j in range(len(all_boxes))]
    if all_keyps is not None:
        cls_keyps_i = [
            _id_or_index(ix, all_keyps[j]) for j in range(len(all_keyps))]
    else:
        cls_keyps_i = None
    if all_tracks is not None:
        cls_tracks_i = [
            _id_or_index(ix, all_tracks[j]) for j in range(len(all_tracks))]
    else:
        cls_tracks_i = None
    pred = _vis_single_frame(
        im.copy(), cls_boxes_i, None, cls_keyps_i, cls_tracks_i, thresh)
    return pred


def main(name_scope, gpu_dev, args):
    im_list = sorted(glob.glob(osp.join(args.im_folder_path + '*.jpg')))
    num_images = len(im_list)
    folder_path = args.im_folder_path
    print(folder_path)
    folder_name=folder_path.split('/')[-1]
    print(folder_name)
    model = initialize_model_from_cfg()
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)

    for i in range(len(im_list)):
        print('Processing Detection for image {}'.format(im_list[i]))
        im_ = cv2.imread(im_list[i])
	assert im_ is not None
        im_ = np.expand_dims(im_, 0)
        with core.NameScope(name_scope):
            with core.DeviceScope(gpu_dev):
                cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
                    model, im_, None)                                        #TODO: Parallelize detection

        extend_results(i, all_boxes, cls_boxes_i)
        if cls_segms_i is not None:
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)


    cfg_yaml = yaml.dump(cfg)

    det_name = folder_name + '_detections.pkl'
    det_file = osp.join(args.out_path, det_name)
    robust_pickle_dump(
        dict(all_boxes=all_boxes,
             all_keyps=all_keyps,
             cfg=cfg_yaml),
        det_file)
    if args.visualize:
     	for i in range(len(im_list)):
	     im_path = im_list[i]
	     im_name = im_path.split('/')[-1].split('.')[0]
#	     if keypoints is not None and len(keypoints) > i:
#    	        vis_im = vis_utils.vis_keypoints(im, keypoints[i], kp_thresh = 2, linewidth=3)
#		cv2.imwrite(osp.join(args.out_path, im_name+'_vis.jpg'), vis_im)
if __name__=='__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parse_args()
    if args.out_path == None:
        args.out_path = args.im_folder_path

    utils.c2.import_custom_ops()
    utils.c2.import_detectron_ops()
    utils.c2.import_contrib_ops()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    assert_and_infer_cfg()
    gpu_dev = core.DeviceOption(caffe2_pb2.CUDA, cfg.ROOT_GPU_ID)
    name_scope = 'gpu_{}'.format(cfg.ROOT_GPU_ID)
    main(name_scope, gpu_dev, args)
