import pdb
import torch
from torchvision import transforms
from os import listdir
from PIL import Image
import json
import random
from PIL import ImageFile

from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import Dataset

class BDD100k_Dataset(Dataset):
    def __init__(self,
                 f,
                 category_list,
                 imgs_path,
                 set='train'):
        import glob
        import os
        self.set = set
        if self.set == 'train':
            self.label_json = 'bdd100k_labels_images_train.json'
            self.images = sorted(glob.glob(os.path.join(imgs_path, self.set) + '/*.jpg'))
        else:
            self.label_json = 'bdd100k_labels_images_val.json'
            self.images = sorted(glob.glob(os.path.join(imgs_path, self.set) + '/*.jpg'))
        self.category_list = category_list
        self.json_file = open(os.path.join(f, self.label_json),'r')
        self.target_files = json.load(self.json_file)
        self.labels = {}
        for image in self.images:
            self.labels[os.path.basename(image)] = extract_json_label(os.path.basename(image),os.path.join(imgs_path,self.set),self.target_files,self.category_list)
            if self.labels[os.path.basename(image)] is None:
                del self.labels[os.path.basename(image)]
                self.images.remove(image)
        # Check to ensure all images have lables:
        for image in self.images:
            try:
                self.labels[os.path.basename(image)]
            except:
                print('Data Checker Error: Image has no existing label')
                self.images.remove(image)

    def __getitem__(self,
                    idx):
        import os
        self.transform = transforms.Compose([
            transforms.Resize((448, 448), Image.NEAREST),
            transforms.ToTensor(),
        ])
        image_p = self.images[idx]
        image = Image.open(image_p)
        image_tensor = self.transform(image)
        label = self.labels[os.path.basename(image_p)]
        return image_tensor, label

    def __len__(self):
        import os
        return len(self.labels)

def extract_json_label(chosen_image,folder_path,target_files,category_list):
    for json_el in target_files:
        if json_el['name'] == chosen_image:
            img_label = json_el
            try:
                if img_label["labels"] is None:  # Checks if a label exists for the given image
                    raise 'NotLabelFound'
            except:
                print('No label for image')
                return None
            target_tensor = transform_label_to_tensor(img_label,
                                                      folder_path,
                                                      category_list)
            return target_tensor

def transform_label_to_tensor(img_label,folder_path,category_list, split_size=7):
    import os
    from PIL import Image
    img = Image.open(os.path.join(folder_path,img_label['name']))
    # Here is the information stored
    target_tensor = torch.zeros(split_size, split_size, 5 + len(category_list))

    for labels in range(len(img_label["labels"])):
        # Store the category index if its contained within the category_list.
        category = img_label["labels"][labels]["category"]
        if category not in category_list:
            continue
        ctg_idx = category_list.index(category)

        # Store the bounding box information and rescale it by the resize factor.
        x1 = img_label["labels"][labels]["box2d"]["x1"] * (448 / img.size[0])
        y1 = img_label["labels"][labels]["box2d"]["y1"] * (448 / img.size[1])
        x2 = img_label["labels"][labels]["box2d"]["x2"] * (448 / img.size[0])
        y2 = img_label["labels"][labels]["box2d"]["y2"] * (448 / img.size[1])

        # Transforms the corner bounding box information into a mid bounding
        # box information
        x_mid = abs(x2 - x1) / 2 + x1
        y_mid = abs(y2 - y1) / 2 + y1
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Size of a single cell
        cell_dim = int(448 / split_size)

        # Determines the cell position of the bounding box
        cell_pos_x = int(x_mid // cell_dim)
        cell_pos_y = int(y_mid // cell_dim)

        # Check if the cell already contains an object
        if target_tensor[cell_pos_y][cell_pos_x][0] == 1:
            continue

        # Stores the information inside the target_tensor
        target_tensor[cell_pos_y][cell_pos_x][0] = 1
        target_tensor[cell_pos_y][cell_pos_x][1] = (x_mid % cell_dim) / cell_dim
        target_tensor[cell_pos_y][cell_pos_x][2] = (y_mid % cell_dim) / cell_dim
        target_tensor[cell_pos_y][cell_pos_x][3] = width / 448
        target_tensor[cell_pos_y][cell_pos_x][4] = height / 448
        target_tensor[cell_pos_y][cell_pos_x][ctg_idx + 5] = 1

    return target_tensor
