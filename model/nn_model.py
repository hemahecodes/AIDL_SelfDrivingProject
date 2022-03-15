####Model design####
import torch.nn as nn
import torch.nn.functional as F


class YoloV1Model(nn.Module):
    def __init__(self, channels=3, classes=13, bb=2, s=7):
        super(YoloV1Model, self).__init__()
        self.c = channels
        self.S = s
        self.classes = classes
        self.bb = bb  # Number of bounding boxes per cell
        # 448x448 input
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                               padding=1)  # 448x448x3 ->224x224x64
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)  # 224x224x64 ->112x112x64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3,
                               padding=1)  # 112x112x64 -> 112x112x192 (192 = 3*64)
        self.batchnorm2 = nn.BatchNorm2d(192)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)  # 112x112x192 -> 56x56x192
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)  # 56x56x192 -> 56x56x128
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # 56x56x128 -> 56x56x256
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)  # 56x56x256 -> 56x56x256
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 56x56x256 -> 56x56x512
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)  # 56x56x512 -> 28x28x512
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)  # 28x28x512 -> 28x28x256
        self.batchnorm7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 28x28x256 -> 28x28x512
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)  # 28x28x512 -> 28x28x256
        self.batchnorm9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 28x28x256 -> 28x28x512
        self.batchnorm10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)  # 28x28x512 -> 28x28x256
        self.batchnorm11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 28x28x256 -> 28x28x512
        self.batchnorm12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)  # 28x28x512 -> 28x28x256
        self.batchnorm13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 28x28x256 -> 28x28x512
        self.batchnorm14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)  # 28x28x512 -> 28x28x512
        self.batchnorm15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)  # 28x28x512 -> 28x28x1024
        self.batchnorm16 = nn.BatchNorm2d(1024)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)  # 28x28x1024 -> 14x14x1024
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)  # 14x14x1024 -> 14x14x512
        self.batchnorm17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,
                                padding=1)  # 14x14x512 -> 14x14x1024
        self.batchnorm18 = nn.BatchNorm2d(1024)
        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)  # 14x14x1024 -> 14x14x512
        self.batchnorm19 = nn.BatchNorm2d(512)
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,
                                padding=1)  # 14x14x512 -> 14x14x1024
        self.batchnorm20 = nn.BatchNorm2d(1024)
        # self.avgpool = nn.AvgPool2d(kernel_size = , stride= , padding= , ceil_mode= , count_include_pad= , divisor_override= )
        # self.fc0 = nn.Linear(in_features = , out_features = )
        # Pretrain until this layer with ImageNet 1000-class classification. 224x224
        self.conv21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3,
                                padding=1)  # 14x14x1024 -> 14x14x1024
        self.batchnorm21 = nn.BatchNorm2d(1024)
        self.conv22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2,
                                padding=1)  # 14x14x1024 -> 7x7x1024
        self.batchnorm22 = nn.BatchNorm2d(1024)
        self.conv23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)  # 7x7x1024 -> 7x7x1024
        self.batchnorm23 = nn.BatchNorm2d(1024)
        self.conv24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)  # 7x7x1024 -> 7x7x1024
        self.batchnorm24 = nn.BatchNorm2d(1024)
        self.flatten = nn.Flatten(),
        self.fc1 = nn.Linear(in_features=1024 * 7 * 7, out_features=496)  # 7x7x1024 -> 4096
        self.fc2 = nn.Linear(in_features=496,
                             out_features=self.S * self.S * (self.bb * 5 + self.classes))  # 4096 -> 7x7x30 = 1470
        self.dropout = nn.Dropout(0.0)

        # Resolution is bumped to 448x448

        self.leakyrelu = nn.LeakyReLU(0.1)  # Leaky RELU Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.maxpool1(self.leakyrelu(self.batchnorm1(self.conv1(x))))  # conv1
        out2 = self.maxpool2(self.leakyrelu(self.batchnorm2(self.conv2(out1))))  # conv2
        out3 = self.maxpool3(self.leakyrelu(self.batchnorm6(self.conv6(self.leakyrelu(self.batchnorm5(self.conv5(
            self.leakyrelu(self.batchnorm4(
                self.conv4(self.leakyrelu(self.batchnorm3(self.conv3(out2)))))))))))))  # From conv3 to conv6
        out4 = self.leakyrelu(self.batchnorm11(self.conv11(self.leakyrelu(self.batchnorm10(self.conv10(self.leakyrelu(
            self.batchnorm9(self.conv9(self.leakyrelu(self.batchnorm8(
                self.conv8(self.leakyrelu(self.batchnorm7(self.conv7(out3)))))))))))))))  # From conv7 to conv11
        out5 = self.maxpool4(self.leakyrelu(self.batchnorm16(self.conv16(self.leakyrelu(self.batchnorm15(self.conv15(
            self.leakyrelu(self.batchnorm14(self.conv14(self.leakyrelu(self.batchnorm13(
                self.conv13(self.leakyrelu(self.batchnorm12(self.conv12(out4))))))))))))))))  # From conv12 to conv16
        out6 = self.leakyrelu(self.batchnorm20(self.conv20(self.leakyrelu(self.batchnorm19(self.conv19(self.leakyrelu(
            self.batchnorm18(
                self.conv18(self.leakyrelu(self.batchnorm17(self.conv17(out5))))))))))))  # From conv17 to conv20
        out7 = self.leakyrelu(self.batchnorm24(self.conv24(self.leakyrelu(self.batchnorm23(self.conv23(self.leakyrelu(
            self.batchnorm22(
                self.conv22(self.leakyrelu(self.batchnorm21(self.conv21(out6))))))))))))  # From conv21 to conv24
        bsz, nch, height, width = out7.shape
        out7 = out7.view(bsz, nch * height * width)
        out = self.fc2(self.leakyrelu(self.dropout(self.fc1(out7))))
        out = self.sigmoid(out)
        # out = self.fc2(self.fc1(out7))
        return out


