import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from model.nn_model import YoloV1Model
import torchvision
import pdb
import wandb
import os
import numpy as np
import argparse as ap
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import Dataset
import glob
from utils import YoloLoss
from data import BDD100k_Dataset
from utils import mAP
import logging
from utils import extract_boxes


def get_args():
    parser = ap.ArgumentParser(description='Code to Train and Evaluate an 2D Object Detection Model (YOLO V1).')
    parser.add_argument("-j", "--json_path", type=str, required=True,
                        help="Path to folder with JSON data.")
    parser.add_argument("-i", "--imgs", type=str, required=True,
                       help= 'Path to foler with images. This path should '
                             'contain 3 folder with val, train and test sets.')
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help='Batch Size. Default is 32.', required=False)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help='Learning Rate. Default is 0.001', required=False)
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help='number of epochs. Default is 100.', required=False)
    args = parser.parse_args()
    return args

def train_net(network, train_loader, eval_loader, optimizer, criterion, hparams, category_list, plot=True):
    """ Function that trains and evals a network for num_epochs,
      showing the plot of losses and accs and returning them.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tr_losses = []
    tr_map = []
    te_losses = []
    te_map = []
    num_epochs = hparams['epochs']
    network.to(device)

    for epoch in range(1, num_epochs + 1):
        logging.info('Running epoch %d', epoch)
        tr_loss, map_training = train_epoch(train_loader, network, optimizer, criterion, hparams, category_list)
        logging.info('Loss training: %d', tr_loss)
        logging.info('mAP training: %d', map_training)
        wandb.log({"Training Loss": tr_loss})
        wandb.log({"Training mAP": map_training})
        te_loss, map_eval = eval_epoch(eval_loader, network, criterion, hparams, category_list)
        logging.info('Loss eval: %d', te_loss)
        logging.info('mAP eval: %d', map_eval)
        wandb.log({"Eval Loss": te_loss})
        wandb.log({"Eval mAP": map_eval})
        te_losses.append(te_loss)
        te_map.append(map_eval)
        tr_losses.append(tr_loss)
        tr_map.append(map_training)
        rets = {'tr_losses':tr_losses, 'te_losses':te_losses, 'map_tr': tr_map, 'map_eval': te_map}
    if plot:
        plt.figure(figsize=(10, 8))
        plt.subplot(2,1,1)
        plt.xlabel('Epoch')
        plt.ylabel('NLLLoss')
        plt.plot(tr_losses, label='train')
        plt.plot(te_losses, label='eval')
        plt.legend()
        plt.subplot(2,1,2)
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.plot(te_map, label='eval')
        plt.plot(tr_map, label='train')
        plt.legend()
        return rets

def eval_epoch(eval_loader, network, loss_fn, hparams,category_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.eval()
    cell_dim = int(448 / 7)
    threshold = 0.5
    test_loss_avg = []
    mean_avg_prec_test = []
    train_idx = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(eval_loader):
            all_pred_boxes = []
            all_target_boxes = []
            images, targets = images.to(device), targets.to(device)
            output = network(images)
            loss = loss_fn(output, targets,set='val')
            predictions = output.reshape(-1, 7, 7,
                                             len(category_list) + 2 * 5)
            pred_boxes = extract_boxes(predictions, len(category_list), 2,
                                       cell_dim, threshold)
            target_boxes = extract_boxes(targets, len(category_list), 1, cell_dim,
                                         threshold)
            for sample_idx in range(len(pred_boxes)):
                nms_boxes = pred_boxes[sample_idx]
                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)

                for box in target_boxes[sample_idx]:
                    all_target_boxes.append([train_idx] + box)
                train_idx += 1
            print('loaded all boxes')
            mean_avg_prec = mAP(all_pred_boxes,
                                all_target_boxes,
                                0.5,
                                category_list)

            test_loss_avg.append(loss.item())
            mean_avg_prec_test.append(mean_avg_prec)


    # Average acc across all correct predictions batches now
    eval_loss = sum(test_loss_avg) / len(test_loss_avg)
    print('Eval set: Average loss: {:.4f}'.format(
        eval_loss
        ))

    return eval_loss, sum(mean_avg_prec_test) / len(mean_avg_prec_test)


def train_epoch(data_train,yolo,optimizer, loss_fn, hparams, category_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss_avg = []
    mean_avg_prec_train = []
    yolo.to(device)
    yolo.train()
    train_idx = 0
    for i, (images, targets) in enumerate(data_train):
        if i >= 500:
            break
        cell_dim = int(448 / 7)
        threshold = 0.5
        all_pred_boxes = []
        all_target_boxes = []
        optimizer.zero_grad()
        img_data = images.to(device)
        target_data = targets.to(device)
        prediction = yolo(img_data)

        loss = loss_fn(prediction,target_data, set='train')
        train_loss_avg.append(loss.item())

        predictions = prediction.reshape(-1, 7, 7,
                                          len(category_list) + 2 * 5)
        pred_boxes = extract_boxes(predictions, len(category_list), 2,
                                   cell_dim, threshold)
        target_boxes = extract_boxes(target_data, len(category_list), 1, cell_dim,
                                     threshold)
        for sample_idx in range(len(pred_boxes)):
            nms_boxes = pred_boxes[sample_idx]
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in target_boxes[sample_idx]:
                all_target_boxes.append([train_idx] + box)
            train_idx += 1
        print('loaded all boxes')
        mean_avg_prec = mAP(all_pred_boxes,
                            all_target_boxes,
                            0.5,
                            category_list)
        mean_avg_prec_train.append(mean_avg_prec)

        loss.backward()
        optimizer.step()

        print('')
        print("=> Saving checkpoint")
        print("")
        torch.save(yolo, 'YOLO_bdd100k_test.pt')
        
    return sum(train_loss_avg) / len(train_loss_avg), sum(mean_avg_prec_train) / len(mean_avg_prec_train)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    wandb.init(project="SelfDriving-project-full-debug", entity="helenamartin",config = {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "optim": 'Adam'
    })
    args = get_args()

    category_list = ["other vehicle", "person", "traffic light", "traffic sign", "truck", "train", "other person",
                     "bus", "car", "rider", "motor", "bike", "trailer"]
    # Defining hyperparameters:
    hparams = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'channels': 3,
        'learning_rate': args.learning_rate,
        'classes': len(category_list),
    }
    jsons_p = args.json_path
    imgs_p = args.imgs
    bddk100k_train = BDD100k_Dataset(jsons_p, category_list, imgs_p)
    bddk100k_eval = BDD100k_Dataset(jsons_p, category_list, imgs_p, set='val')
    yolo = YoloV1Model(channels=hparams['channels'],
                       classes=hparams['classes'],
                       bb=2,
                       s=7)

    data_train = torch.utils.data.DataLoader(
        bddk100k_train,
        batch_size=hparams['batch_size'],
        shuffle=False
    )
    data_eval = torch.utils.data.DataLoader(
        bddk100k_eval,
        batch_size=hparams['batch_size'],
        shuffle=False
    )
    print('Len test dataset:', len(data_train))
    print('Len eval dataset:', len(data_eval))
    optimizer = torch.optim.Adam(params=yolo.parameters(), lr=hparams['learning_rate'], weight_decay=0.0005)
    loss_fn = YoloLoss(C=hparams['classes'], S=7)
    train_net(yolo, data_train, data_eval, optimizer, loss_fn,hparams, category_list)