import torch
from torchvision.transforms.functional import to_tensor
import os
import json
from PIL import Image, ImageDraw

#If we have GPU, we will use it. Otherwise, not.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#We define a class for Berkeley Deep Driving dataset
class DeepDrivingDataset(object):
    label2idx = {"other vehicle": 0,"person": 1,"traffic light": 2,"traffic sign": 3,"truck": 4,"train": 5,"other person": 6,"bus": 7,"car": 8,"rider": 9, "motor": 10, "bike": 11, "trailer": 12}
    def __init__(self, train = True):
        # load all image files, sorting them to
        # ensure that they are aligned
        self.train = train
        if self.train:
          self.img_dir = os.path.join("data","DeepDriving","train") #use os.path.join
        else:
          self.img_dir = os.path.join("data","DeepDriving","val")
        json_file = os.path.join(self.img_dir, "labels_TL.json")
        with open(json_file) as f:
          imgs_anns = json.load(f)

        self.imgs = []
        self.annotations = []
        for idx, v in enumerate(imgs_anns.values()):
          filename = os.path.join(self.img_dir, v["name"])
          self.imgs.append(filename)
          self.annotations.append(v["labels"])

    def __getitem__(self, idx):
        # load images
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")

        # get bounding box coordinates for each object detected
        boxes = []
        categories = []
        for labels in self.annotations[idx]:
          if 'box2d' in labels:
            annotation = labels['box2d']
            lab = labels['category']
            categories.append(self.label2idx[lab])
            #select the corners of the boxes for each axis. it should be a list with 4 values: 2 coordinates.
            boxes.append([annotation["x1"],annotation["y1"],annotation["x2"],annotation["y2"]]) 
          else:
            continue
          
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
        boxes.to(device)
        labels = torch.tensor(categories, dtype=torch.int64, device=device)
        labels.to(device)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        img = to_tensor(img).to(device)
        
        return img, target

    def __len__(self):
        return len(self.imgs)
def collate_fn(batch):
    images = []
    targets = []
    for i, t in batch:
        images.append(i)
        targets.append(t)
    return images, targets
