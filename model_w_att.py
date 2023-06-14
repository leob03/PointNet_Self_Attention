import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model import Tnet
from attention import SA_Layer

class Transform_with_attention(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       
        self.bn_a_1 = nn.BatchNorm1d(3)
        self.bn_a_2 = nn.BatchNorm1d(64)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # self.sa1 = SA_Layer(3)
        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(64)
        self.sa4 = SA_Layer(64)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)
        # xb1 = self.bn_a_1(xb)
        # print(xb.shape)
        # xb = self.sa1(xb)                                                               #layer norm and self-attention layer
        # xb = xb + xb1_i                                                                 #residual
        xb = F.relu(self.bn1(self.conv1(xb)))

        # xb = self.sa1(xb)
        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)
        xb1 = self.sa1(self.bn_a_2(xb))                                                   #layer norm and self-attention layer
        xb2 = self.sa2(xb1)                                                               #stacked attention
        xb3 = self.sa3(xb2)                                                               #stacked attention
        xb4 = self.sa4(xb3)                                                               #stacked attention
        xb = xb + xb4                                                                     #residual
        xb_2 = F.relu(self.bn2(self.conv2(xb)))
        xb_2i = self.bn3(self.conv3(xb_2))
        xb = torch.nn.functional.interpolate(xb.transpose(1,2), scale_factor=16, mode="nearest").transpose(1,2)
        xb = xb + xb_2i

        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet_with_transform(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform_with_attention()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64