from __future__ import division

import os
from Models.Hourglass.pose.detection.util import *
from Models.Hourglass.pose.detection.darknet import Darknet
from Models.Hourglass.pose.detection.preprocess import prep_image

confidence = 0.5
nms_thesh = 0.4
num_classes = 80

cfg_file = "Models/Hourglass/pose/detection/cfg/yolov3.cfg"
file = os.path.join(os.getcwd(), cfg_file)
print(file)
fp = open(file, "r")
names = fp.read().split("\n")[:-1]

weights_file = "./Models/Hourglass/pose/detection/yolov3.weights"
classes = load_classes("./Models/Hourglass/pose/detection/data/coco.names")

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers = num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5, 5) for x in range(num_layers)])
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)

def find_person(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    if label == "person":
        cv2.rectangle(img, c1, c2, [255,255, 0], 1)
    return img

def get_test_input(input_dim, CUDA):
    img = cv2.imread("./dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_

def detect_person(im_file):
    # im_file = "./imgs/messi.jpg"
    bbox = []

    CUDA = torch.cuda.is_available()
    # Set up the neural network
    print("Loading network.....")
    # configure file
    model = Darknet(cfg_file)
    # weight file
    model.load_weights(weights_file)
    print("Network successfully loaded")

    # Input resolution of the network
    model.net_info["height"] = 416
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    # Detection phase
    batches = list(prep_image(im_file, inp_dim))
    print(batches[2])
    im_batches = batches[0]
    im_dim_list = batches[2]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    # model(get_test_input(inp_dim, CUDA), CUDA)

    # for batch in im_batches:
    batch = im_batches
    # load the image
    if CUDA:
        batch = batch.cuda()

    # Apply offsets to the result predictions
    # Tranform the predictions as described in the YOLO paper
    # flatten the prediction vector
    # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
    # Put every proposed box as a row.
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)

    #        prediction = prediction[:,scale_indices]

    # get the boxes with object confidence > threshold
    # Convert the cordinates to absolute coordinates
    # perform NMS on these boxes, and save the results
    # I could have done NMS and saving seperately to have a better abstraction
    # But both these operations require looping, hence
    # clubbing these ops in one loop instead of two.
    # loops are slower than vectorised operations.
    prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)
    output = prediction

    try:
        output
    except NameError:
        print("No detections were made")
        return bbox

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    # img = cv2.imread(im_file)
    for i in range(len(output)):
        if output[i][-1] == 0.0:
            x1 = output[i][1]
            y1 = output[i][2]
            x2 = output[i][3]
            y2 = output[i][4]
            bbox = [x1, y1, x2, y2]
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
    #     else:
    #         print("error")
    # cv2.imshow("test", img)
    # cv2.waitKey(0)

    return bbox





