"""Apply Transfer learning in a pre-trained Faster-RCNN for Self-Driving Purposes
"""

#Importing needed packages
import torch
import torchvision
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
import argparse as ap
from dataset import DeepDrivingDataset
from dataset import collate_fn

#If we have GPU, we will use it. Otherwise, not.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-b", "--backb", type=int, required=True,
                        help="Which backbone do you want to use? 1. MobileNetV2; 2. ResNet-50; 3. Efficient Net_b7")
    parser.add_argument("-c", "--num_classes", type=int, required=True,
                       help="Number of classes")
    parser.add_argument("-e", "--num_epochs", type=int, required=True,
                       help="Number of epochs for training the model")
    args = parser.parse_args()
    return args

#Replace the classifier with a new one, that has num_classes which is user-defined (in our case, 11)
# num_classes = 11  #{"person": 0,"rider": 1,"car": 2,"truck": 3,"bus": 4,"train": 5,"motorcycle": 6,"bike": 7,"traffic light": 8,"traffic sign": 9, "motor": 10}

# The backbone refers to the network which takes as input the image and extracts the feature map upon which the rest of the network is based
# So, in our case, we input the chosen backbone as an argument.
args = get_args()
backb = args.backb
num_classes = args.num_classes
num_epochs = args.num_epochs
if backb == 1:
  backbone = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
  backb_used = "MobileNetV3"
elif backb == 2:
  backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  backb_used = "ResNet 50"
elif backb == 3:
  backbone = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
  backb_used = "MobileNetV3-320"

# Now, we make the RPN generate 5x3 anchors per spatial location (5 different sizes, 3 different aspect ratios)
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)

# It's time to define our model!
#Function that will give us the model
def get_model_object_detection(num_classes):
    # load an object detection model pre-trained on COCO
    model = backbone
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model 

# Get the model using our helper function
model = get_model_object_detection(num_classes)
model.to(device) # move model to the right device
training_dataset = DeepDrivingDataset(train=True)
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=4, num_workers=0, collate_fn=collate_fn)

# Optimizer used: SGD
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# LR scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(num_epochs):
        # train for one epoch, printing every 2 iterations
        model.train()
        for i, (images, targets) in enumerate(training_dataloader):
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

            if i%2 == 0:
                loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
                print(f"[{i}/{len(training_dataloader)}] loss: {loss_dict_printable}")
        # update the learning rate
        lr_scheduler.step()

model_name = "models/Pretrained FasterRCNN with " + backb_used+".pth"
torch.save({'model_state_dict': model.state_dict()}, model_name)
