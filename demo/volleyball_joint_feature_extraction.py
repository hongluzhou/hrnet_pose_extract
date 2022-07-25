from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

import sys
sys.path.append("../lib")
import time

# import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform

import pdb
import pickle  
import glob  
from matplotlib import pyplot as plt  
py_min, py_max = min, max
from tqdm import tqdm


CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_person_detection_boxes(model, img, threshold=0.5):
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    pred = model([transformed_img.to(CTX)])  # Pass the image to the model
    # Use the first detected person
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

    person_boxes = []
    # Select box has score larger than threshold and is person
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, _ = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(root_dir, video_id, clip_id):
    save_folder = os.path.join(root_dir, video_id)
    os.makedirs(save_folder, exist_ok=True)
    return os.path.join(save_folder, clip_id+'.pickle')
    


def parse_args():
    parser = argparse.ArgumentParser(description='HRNet inference on Volleyball')
    parser.add_argument('--cfg', type=str, default='inference-config-hrnet_w32_256x192.yaml')
    parser.add_argument('--model_file', type=str, default='../models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth')  
    
    parser.add_argument('--dataset_path', type=str, default='/home/honglu/Dataset/volleyball/videos')
    parser.add_argument('--track_path', type=str, default='/home/honglu/Dataset/volleyball/tracks_normalized.pkl')
    parser.add_argument('--save_path', type=str, default='/home/honglu/Dataset/volleyball/joints')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    OW = 720
    OH = 1280
    
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args) 
    if os.path.exists(args.save_path) and os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
    
    with open(args.track_path, 'rb') as f:
        tdata = pickle.load(f) 
     
    
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('=> loading model from {}'.format(args.model_file))
        pose_model.load_state_dict(torch.load(args.model_file), strict=False)

    pose_model.to(CTX)
    pose_model.eval()
    
    videos = [file for file in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, file))]


    
    # Start
    # --------
    for video_id in tqdm(videos):
        
        clips = [file for file in os.listdir(os.path.join(args.dataset_path, video_id)) if os.path.isdir(os.path.join(args.dataset_path, video_id, file))]

        for clip_id in clips:
            
            joint_output = dict()
            
            for frame_id in sorted(tdata[(int(video_id), int(clip_id))]): 

                img_path = args.dataset_path + '/{}/{}/'.format(video_id, clip_id) + str(frame_id) + '.jpg' 
                image_bgr = cv2.imread(img_path)   

                if image_bgr is None:
                    print('{} is None!'.format(img_path))
                    continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # Clone 1 image for pose estimation
                if cfg.DATASET.COLOR_RGB: 
                    image_pose = image_rgb.copy()
                else: 
                    image_pose = image_bgr.copy()
                    

                # Load object detection box 
                pred_boxes = []

                for track_id, box in enumerate(tdata[(int(video_id), int(clip_id))][frame_id]):

                    y1,x1,y2,x2 = box
                    box = [(int(x1*OH), int(y1*OW)), (int(x2*OH), int(y2*OW))] 

                    pred_boxes.append(box)


                # Can not find people. Move to next frame
                if not pred_boxes: 
                    print('pred_boxes is empty!')
                    pdb.set_trace()  # Not possible

                
                # pose estimation : for multiple people
                centers = []
                scales = []
                for box in pred_boxes:
                    center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                    centers.append(center)
                    scales.append(scale)

                # now = time.time()
                pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)
                # then = time.time()
                # print("Find person pose in: {} sec".format(then - now))

                # ========
                # Save data     
                this_frame_all_people = []
                for track_id in range(len(pose_preds)):  # Loop over person
                    
                    this_person = []
                    coords = pose_preds[track_id]
                    
                    for k in range(len(coords)):  # Loop over joints
                        coord = coords[k]
                        x_coord, y_coord = coord[0], coord[1]
                        
                        this_person.append([x_coord, y_coord, k])  # [x, y, class_id] see COCO_KEYPOINT_INDEXES
                        
                    this_frame_all_people.append(this_person)
                         
                
                joint_output[frame_id] = np.array(this_frame_all_people)

            # Save data of this clip
            save_dir = prepare_output_dirs(args.save_path, video_id, clip_id)  ########
            with open(save_dir, 'wb') as f:
                pickle.dump(joint_output, f)
            
            


if __name__ == '__main__':
    main()
