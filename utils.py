from random import randint
import torch
import torch.nn as nn
import wandb

def IoU(target, prediction):
    """
    Calculates the Intersection over Union of two bounding boxes.

    Parameters:
    target (list): A list with bounding box coordinates in the corner format.
    predictions (list): A list with bounding box coordinates in the corner format.

    Returns:
    iou_value (float): The score of the IoU over the two boxes.
    """

    # Calculate the corner coordinates of the intersection
    i_x1 = max(target[0], prediction[0])
    i_y1 = max(target[1], prediction[1])
    i_x2 = min(target[2], prediction[2])
    i_y2 = min(target[3], prediction[3])

    intersection = max(0, (i_x2 - i_x1)) * max(0, (i_y2 - i_y1))
    union = ((target[2] - target[0]) * (target[3] - target[1])) + ((prediction[2] - prediction[0]) *
                                                                   (prediction[3] - prediction[1])) - intersection
    if union == 0 or intersection == 0:
        return 0
    iou_value = intersection / union
    return iou_value


def MidtoCorner(mid_box, cell_h, cell_w, cell_dim):
    """
    Transforms bounding box coordinates which are in the mid YOLO format into the
    common corner format with the correct pixel locations.

    Parameters:
        mid_box (list): Bounding box coordinates which are in the mid YOLO format
        [x_mid, y_mid, width, height].
        cell_h (int): Height index of the cell with the bounding box.
        cell_w (int): Width index of the cell with the bounding box.
        cell_dim (int): Dimension of a single cell.

    Returns:
        corner_box (list): A list containing the coordinates of the bounding
        box in the common corner format [x1, y2, x2, y2].
    """

    # Transform the coordinates from the YOLO format into normal pixel values
    centre_x = mid_box[0] * cell_dim + cell_dim * cell_w
    centre_y = mid_box[1] * cell_dim + cell_dim * cell_h
    width = mid_box[2] * 448
    height = mid_box[3] * 448

    # Calculate the corner values of the bounding box
    x1 = int(centre_x - width / 2)
    y1 = int(centre_y - height / 2)
    x2 = int(centre_x + width / 2)
    y2 = int(centre_y + height / 2)

    corner_box = [x1, y1, x2, y2]
    return corner_box


def extract_boxes(yolo_tensor, num_classes, num_boxes, cell_dim, threshold):
    """
    Extracts all bounding boxes from a given tensor and transforms them into a list.
    Removes all bounding boxes which have a confidence score smaller than the
    specified threshold.

    Parameters:
        yolo_tensor (tensor): The tensor from which the bounding boxes need to
        be extracted.
        num_classes (int): Amount of classes which are being predicted.
        num_boxes (int): Amount of bounding boxes which are being predicted.
        cell_dim (int): Dimension of a single cell.
        threshold (float): Threshold for the confidence score of predicted
        bounding boxes.

    Returns:
        all_bboxes (list): A list where each element is a list representing one
        image from the batch. This inner list contains other lists which represent
        the bounding boxes within this image.
        The box lists are specified as [class_pred, conf_score, x1, y1, x2, y2]
    """

    all_bboxes = []  # Stores the final output

    for sample_idx in range(yolo_tensor.shape[0]):
        bboxes = []  # Stores all bounding boxes of a single image
        for cell_h in range(yolo_tensor.shape[1]):
            for cell_w in range(yolo_tensor.shape[2]):

                # Used to extract the bounding box with the highest confidence
                best_box = 0
                max_conf = 0.
                for box_idx in range(num_boxes):
                    if yolo_tensor[sample_idx, cell_h, cell_w, box_idx * 5] > max_conf:
                        max_conf = yolo_tensor[sample_idx, cell_h, cell_w, box_idx * 5]
                        best_box = box_idx
                conf = yolo_tensor[sample_idx, cell_h, cell_w, best_box * 5]
                if conf < threshold:
                    continue

                # Used to extract the class with the highest score
                best_class = 0
                max_conf = 0.
                for class_idx in range(num_classes):
                    if yolo_tensor[sample_idx, cell_h, cell_w, num_boxes * 5 + class_idx] > max_conf:
                        max_conf = yolo_tensor[sample_idx, cell_h, cell_w, num_boxes * 5 + class_idx]
                        best_class = class_idx

                cords = MidtoCorner(yolo_tensor[sample_idx, cell_h, cell_w,
                                    best_box * 5 + 1:best_box * 5 + 5], cell_h, cell_w, cell_dim)
                x1 = cords[0]
                y1 = cords[1]
                x2 = cords[2]
                y2 = cords[3]

                bboxes.append([best_class, conf, x1, y1, x2, y2])
        all_bboxes.append(bboxes)
    return all_bboxes


def mAP(pred_boxes,true_boxes,iou_threshold,category_list):
    from collections import Counter
    # list storing all AP for respective classes
    average_precisions = []

    epsilon = 1e-6

    for c in range(len(category_list)):
        print('Calculating average precision for class: ' + str(category_list[c]))
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = IoU(
                    gt[3:], detection[3:]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    print(average_precisions)
    return sum(average_precisions) / len(average_precisions)


class YoloLoss(nn.Module):
    def __init__(self, S=14, B=2,
                 C=80, set='train'):  # S is the number of grids in which we are going to divide (7x7), B is the quantity of boundig box per cell, C is the number of classes
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.set = set

    def forward(self, predictions, target, set):
        num_classes = self.C
        predictions = predictions.reshape(-1, self.S, self.S,
                                          self.C + self.B * 5)  # Make sure that the shape is (-1,7,7,80+10) = (-1,7,7,90)
        iou_b1 = self.intersection_over_union(predictions[..., num_classes+1:num_classes+1+4], target[...,
                                                                  num_classes+1:num_classes+1+4])  # From 0 to 79 is for class probabilities, 80 i for class score
        iou_b2 = self.intersection_over_union(predictions[..., num_classes+6:num_classes+10], target[..., num_classes+1:num_classes+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., num_classes].unsqueeze(3)
        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                    bestbox * predictions[..., num_classes+6:num_classes+10]
                    + (1 - bestbox) * predictions[..., num_classes+1:num_classes+5]
            )
        )

        box_targets = exists_box * target[..., num_classes+1:num_classes+5]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
                bestbox * predictions[..., num_classes+5:num_classes+6] + (1 - bestbox) * predictions[..., num_classes:num_classes+1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., num_classes:num_classes+1]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        # no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        # )

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., num_classes:num_classes+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., num_classes:num_classes+1], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., num_classes+5:num_classes+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., num_classes:num_classes+1], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :num_classes], end_dim=-2, ),
            torch.flatten(exists_box * target[..., :num_classes], end_dim=-2, ),
        )

        loss = (
                self.lambda_coord * box_loss  # first two rows in paper
                + object_loss  # third row in paper
                + self.lambda_noobj * no_object_loss  # forth row
                + class_loss  # fifth row
        )

        return loss

    def intersection_over_union(self,boxA, boxB, box_format="midpoint"):
        """
        Calculates intersection over union
        Parameters:
            boxA (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxB (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        Returns:
            tensor: Intersection over union for all examples
        """
        if box_format == "midpoint":
            box1_x1 = boxA[..., 0:1] - boxA[..., 2:3] / 2
            box1_y1 = boxA[..., 1:2] - boxA[..., 3:4] / 2
            box1_x2 = boxA[..., 0:1] + boxA[..., 2:3] / 2
            box1_y2 = boxA[..., 1:2] + boxA[..., 3:4] / 2
            box2_x1 = boxB[..., 0:1] - boxB[..., 2:3] / 2
            box2_y1 = boxB[..., 1:2] - boxB[..., 3:4] / 2
            box2_x2 = boxB[..., 0:1] + boxB[..., 2:3] / 2
            box2_y2 = boxB[..., 1:2] + boxB[..., 3:4] / 2

        if box_format == "corners":
            box1_x1 = boxA[..., 0:1]
            box1_y1 = boxA[..., 1:2]
            box1_x2 = boxA[..., 2:3]
            box1_y2 = boxA[..., 3:4]  # (N, 1)
            box2_x1 = boxB[..., 0:1]
            box2_y1 = boxB[..., 1:2]
            box2_x2 = boxB[..., 2:3]
            box2_y2 = boxB[..., 3:4]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        # .clamp(0) is for the case when they do not intersect
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + 1e-6)