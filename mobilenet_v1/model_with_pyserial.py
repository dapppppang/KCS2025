import torch
import torch.nn as nn
from pyserial_demo.pyserial_demo2 import uart_setup, send_weight, send_tensor, receive_data


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)      # already done by FPGA
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.depthwise(x)      # already done by FPGA
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class MobileNetV1_with_pyserial(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV1_with_pyserial, self).__init__()
        # MobileNetV1 레이어 정의
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Depthwise Separable Convolutions
        self.conv2 = DepthwiseSeparableConv(32, 64, stride=1)
        self.conv3 = DepthwiseSeparableConv(64, 128, stride=2)
        self.conv4 = DepthwiseSeparableConv(128, 128, stride=1)
        self.conv5 = DepthwiseSeparableConv(128, 256, stride=2)
        self.conv6 = DepthwiseSeparableConv(256, 256, stride=1)
        self.conv7 = DepthwiseSeparableConv(256, 512, stride=2)

        # 반복되는 512 레이어
        self.conv8 = DepthwiseSeparableConv(512, 512, stride=1)
        self.conv9 = DepthwiseSeparableConv(512, 512, stride=1)
        self.conv10 = DepthwiseSeparableConv(512, 512, stride=1)
        self.conv11 = DepthwiseSeparableConv(512, 512, stride=1)
        self.conv12 = DepthwiseSeparableConv(512, 512, stride=1)

        # 최종 1024 레이어
        self.conv13 = DepthwiseSeparableConv(512, 1024, stride=2)
        self.conv14 = DepthwiseSeparableConv(1024, 1024, stride=1)

        # 평균 풀링 및 최종 분류 레이어
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 초기 Conv 레이어
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # UART PORT OPEN
        ser = uart_setup()

        # SEND INSTRUCTION & WEIGHT TENSOR
        ser.write(b'10101000')  # SEND INSTRUCTION
        send_weight(ser, 'weight_binary_files/fp32/dwcv2_weight_bin.bin')  # SEND WEIGHT BINARY STRING DATA

        # SEND INSTRUCTION & OUTPUT TENSOR
        ser.write(b'00100000')
        send_tensor(ser, x, torch.float32)

        # RECEIVE RESULT OF FPGA
        x = receive_data(ser, (1, 32, 32, 32), torch.float32)

        # Depthwise Separable Conv 레이어들
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        # 반복되는 512 레이어들
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)

        # 최종 Conv 레이어
        x = self.conv13(x)
        x = self.conv14(x)

        # 평균 풀링
        x = self.avg_pool(x)

        # 벡터화 및 FC 레이어로 연결
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
