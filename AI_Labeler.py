import cv2
import numpy as np

def OpenCV_Model(file, model):
    # Specify the paths for the 2 files
    protoFile = "models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "models/pose/mpi/pose_iter_160000.caffemodel"

    # Read the network into Memory
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Read image
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    # frame = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    # Specify the input image dimensions
    inHeight = 368
    inWidth = int((inHeight/imgHeight)*imgWidth)

    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []
    print(output.shape[1])
    for i in range(15):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (imgWidth * point[0]) / W
        y = (imgHeight * point[1]) / H

        if prob > 0:
            cv2.circle(img, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(img, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                        lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    POSE_PAIRS = {(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(8,9),(9,10),(11,12),(12,13),(8,14),(11,14)}
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img, points[partA], points[partB], (0, 255, 0), 3)

    # imS = cv2.resize(frame, (int(680/frameHeight*frameWidth),680))
    # cv2.imshow("Output-Keypoints", imS)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(points[0:15])
    points = np.asarray(points[0:15])

    # save points as pkl file with model name
    return img, points

def DetectAndTrack_Model(resource, model):
    print("test")
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parse_args('--cfg configs/video/2d_best/01_R101_best_hungarian-4GPU.yaml \
         --img_fol MPII_images/ \
         --output outputs/ \
         TEST.WEIGHTS pretrained_models/configs/video/2d_best/01_R101_best_hungarian-4GPU.yaml/model_final.pkl')
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
