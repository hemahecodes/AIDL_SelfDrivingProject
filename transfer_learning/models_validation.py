# Main file of the project
import pdb
import wandb
import torch
import os
import numpy as np
import argparse as ap
import torchvision
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import Dataset
import glob
from utils import YoloLoss
import math
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
from torchvision import transforms


use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-m", "--model_used", type=int, required=True,help="Model to check.")
    parser.add_argument("-jv", "--json_path_val", type=str, required=True,help="Path to folder with JSON data.")
    parser.add_argument("-iv", "--imgs_val", type=str, required=True,help="Path to folder with images.")
    args = parser.parse_args()
    return args

def create_model(num_classes, model_used):

    #Define the backbone used:
    if model_used == 1:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        model_name = "models/Pretrained FasterRCNN with MobileNetV3.pth"
    elif model_used == 2:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model_name = "models/Pretrained FasterRCNN with ResNet 50.pth"
    elif model_used == 3:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        model_name = "models/Pretrained FasterRCNN with MobileNetV3 320.pth"
        
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model_name, model
    
args = get_args()
model_used = args.model_used
model_name, model = create_model(num_classes=13, model_used = model_used)
model.to(device)

checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

iou_threshold = 0.2
score_threshold = 0.4
label2idx = {"other vehicle": 0,"person": 1,"traffic light": 2,"traffic sign": 3,"truck": 4,"train": 5,"other person": 6,"bus": 7,"car": 8,"rider": 9, "motor": 10, "bike": 11, "trailer": 12}
idx2label = {v: k for k, v in label2idx.items()}

def eval_mod(jsons_p_val, imgs_p_val):
    # Training yolo v1
    import torch
    from torchsummary import summary
    import matplotlib.pyplot as plt
    # from model.nn_model import YoloV1Model
    from data import DataLoader
    from utils import retrieve_box
    import torchvision
    from torchvision.utils import draw_bounding_boxes

    category_list = ["other vehicle", "person", "traffic light", "traffic sign",
                     "truck", "train", "other person", "bus", "car", "rider", "motor",
                     "bike", "trailer"]


    # Defining hyperparameters:
    hparams = {
        'num_epochs': 1,
        'batch_size': 5,
        'channels': 3,
        'learning_rate': 0.0001,
        'classes': len(category_list),
        'nsamples': 25000,
        'grid_size': 14
    }
    use_gpu = True
    data_test = \
        DataLoader(
            img_files_path=imgs_p_val,
            target_files_path=jsons_p_val,
            category_list=category_list,
            split_size=14, # Amount of grid cells
            batch_size=hparams['batch_size'],
            load_size=1
        )
    
    # Move model to the GPU
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    print(device)
    # loss_fn = YoloLoss(C=hparams['classes'], S =hparams['grid_size'])

    for epoch in range(hparams['num_epochs']):

        total_AP = []
        test_loss_avg = []
        print("")
        data_test.LoadFiles()  # Resets the Validation DataLoader for a new epoch
        
        model.eval()
        print("DATA IS BEING VALIDATED FOR A NEW EPOCH")
        print("")
        while len(data_test.img_files) > 0:

            print("LOADING NEW BATCHES TO VALIDATE")
            print("Remaining files:" + str(len(data_test.img_files)))
            print("")
            data_test.LoadData()  # Loads new batches
            

            for batch_idx, (img_data, target_data) in enumerate(data_test.data):
                if batch_idx > hparams['nsamples']:
                    break
                img_data = img_data.to(device) #Image loaded, converted as a tensor and resized to 448x448
                target_data = target_data.to(device) #Labels
                prediction = model(img_data)
                epsilon = 1e-6
                cell_dim = 448/hparams['grid_size'] #Dimension of each cell in which the image is divided

                for i in range(hparams['batch_size']):
                    precisions = [0]*len(category_list)
                    recalls = [0]*len(category_list)
                    im_AP = []
                    w = 1280
                    h = 720
                    newsize = (w, h)
                    im = to_pil_image(img_data[i])
                    im = im.resize(newsize)
                    draw = ImageDraw.Draw(im)
                    classes_target = []
                    boxes_target = []

                    #Now we go cell by cell to obtain all the bounding boxes and classes of the ground truth
                    for j in range(hparams['grid_size']):
                        for k in range(hparams['grid_size']):
                            if int(target_data[i][j][k][0].tolist()) == 1: 
                                box_target = target_data[i][j][k][1:5].tolist() 
                                class_target = torch.argmax(target_data[i][j][k][5:]).tolist()

                                #Undoing the transformations done on the data loader to obtain the format [xmin,ymin,xmax,ymax] again
                                x_c = box_target[0]*cell_dim+k*cell_dim
                                y_c = box_target[1]*cell_dim+j*cell_dim
                                w1 = box_target[2]*448
                                h1 = box_target[3]*448
                                x1 = (x_c-w1/2)*w/448
                                x2 = (x_c+w1/2)*w/448
                                y1 = (y_c-h1/2)*h/448
                                y2 = (y_c+h1/2)*h/448
                                box_target = torch.Tensor([x1,y1,x2,y2])
                                boxes_target.append(box_target)
                                classes_target.append(class_target)
                    total_boxes_target = len(boxes_target) #Total quantity of bboxes on the GT
                    true_boxes_used = torch.zeros(total_boxes_target) #We will be checking each bbox used (used means compared with a bbox detected)
                    true_boxes_counted = torch.zeros(total_boxes_target) #Needed for defining the total number of bbox of a specific class in GT
                    keep_idx = torchvision.ops.nms(prediction[i]['boxes'], prediction[i]['scores'], iou_threshold) #Performs non-maximum suppression (NMS) on the boxes according to their IoU
                    #We keep only the predicted bboxes, sxores and labels that we obtain after NMS
                    boxes = [b for i, b in enumerate(prediction[i]["boxes"]) if i in keep_idx] 
                    scores = [s for i, s in enumerate(prediction[i]["scores"]) if i in keep_idx]
                    labels = [l for i, l in enumerate(prediction[i]["labels"]) if i in keep_idx]
                    #Loop by classes in order to compute TP, FP, recall, precision per class
                    for c in range(len(category_list)):
                        boxes_pred = []
                        scores_pred = []
                        for l in range(len(boxes)):
                            if labels[l] == c and scores[l] > score_threshold:
                                #Resizing the predictions so they are not on images (448,448) but on the real size
                                x1_pred = boxes[l][0]*w/448
                                x2_pred = boxes[l][2]*w/448
                                y1_pred = boxes[l][1]*h/448
                                y2_pred = boxes[l][3]*h/448
                                box_pred = torch.Tensor([x1_pred,y1_pred, x2_pred, y2_pred])
                                boxes_pred.append(box_pred)
                                scores_pred.append(scores[l])
                        
                        #Each prediction will be a True Positive or a False Positive
                        TP = torch.zeros((len(boxes_pred)))
                        FP = torch.zeros((len(boxes_pred)))
                        total_boxes_target_class = 0

                        #We loop over the boxes predicted
                        for det_idx, p in enumerate(boxes_pred):
                            iou_max = 0
                            #For each box predicted, we will look for the best (highest IoU) GT box and then GT box will be checked as used.
                            for idx, t in enumerate(boxes_target):
                                if classes_target[idx] == c:
                                    if true_boxes_counted[idx] == 0:
                                        total_boxes_target_class = total_boxes_target_class + 1
                                        true_boxes_counted[idx] = 1
                                    x1 = torch.max(t[0], p[0])
                                    y1 = torch.max(t[1], p[1])
                                    x2 = torch.min(t[2], p[2])
                                    y2 = torch.min(t[3], p[3])
                                    # .clamp(0) is for the case when they do not intersect
                                    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

                                    box1_area = abs((t[2] - t[0]) * (t[3] - t[1]))
                                    box2_area = abs((p[2] - p[0]) * (p[3] - p[1]))

                                    iou = intersection / (box1_area + box2_area - intersection + 1e-6)

                                    if iou >= iou_max:
                                        iou_max = iou
                                        true_index = idx
                            #If the maximum IoU is greater than the threshold and the GT bbox is not used yet, we have a TP
                            if iou_max > iou_threshold:
                                if true_boxes_used[idx] == 0:
                                    TP[det_idx] = 1
                                    true_boxes_used[true_index] = 1
                                    coords = p.cpu().tolist()
                                    draw.rectangle(coords, width = 3) 
                                    text = f"{idx2label[c]} {scores_pred[det_idx]*100:.2f}%"
                                    draw.text([coords[0], coords[1]-15], text)
                                else:
                                    FP[det_idx] = 1
                                    coords = p.cpu().tolist()
                                    draw.rectangle(coords, width = 3) 
                                    text = f"{idx2label[c]} {scores_pred[det_idx]*100:.2f}%"
                                    draw.text([coords[0], coords[1]-15], text)
                            else:
                                FP[det_idx] = 1
                                coords = p.cpu().tolist()
                                draw.rectangle(coords, width = 3) 
                                text = f"{idx2label[c]} {scores_pred[det_idx]*100:.2f}%"
                                draw.text([coords[0], coords[1]-15], text)
                        if total_boxes_target_class == 0:
                            continue
                        else:
                            #Sum of all TP and FP to compute recall and precision for each class
                            TP_cumsum = torch.cumsum(TP, dim = 0)
                            FP_cumsum = torch.cumsum(FP, dim = 0)
                            recalls = TP_cumsum / (total_boxes_target_class + epsilon)
                            precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
                            precisions = torch.cat((torch.tensor([1]), precisions))
                            recalls = torch.cat((torch.tensor([0]), recalls))
                            #Average precision is the area under the curve of the precision-recall (we approximate that with trapezoide rule)
                            im_AP.append(torch.trapz(precisions, recalls))
                    print("Average precision of this image: ", sum(im_AP) / len(im_AP))
                    im.show()
                    total_AP.append(sum(im_AP) / len(im_AP))
        
        print("Mean Average precision of this epoch: ", np.mean(total_AP) )                        


if __name__ == '__main__':
    args = get_args()
    model_used = args.model_used
    jsons_p_val = args.json_path_val
    imgs_p_val = args.imgs_val
    eval_mod(jsons_p_val, imgs_p_val)


