import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_utils import *
from .FlowNet_model import FlowNet
from .Backward_warp_layer import Backward_warp


class MultiscaleWarpingNet(nn.Module):

    def __init__(self):
        super(MultiscaleWarpingNet, self).__init__()
        self.FlowNet = FlowNet(2)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(1)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, input1_HR, input2_bic_SR):
        # input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        # input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
        # input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()

        # input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
        # input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()
        # input_img2_SR = torch.from_numpy(buff['input_img2_SR']).cuda()

        flow = self.FlowNet(input2_bic_SR, input1_HR)

        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input2_bic_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input1_HR)

        warp_21_conv1 = self.Backward_warp(HR2_conv1, flow_12_1)
        warp_21_conv2 = self.Backward_warp(HR2_conv2, flow_12_2)
        warp_21_conv3 = self.Backward_warp(HR2_conv3, flow_12_3)
        warp_21_conv4 = self.Backward_warp(HR2_conv4, flow_12_4)

        sythsis_output = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1, warp_21_conv2,
                                             warp_21_conv3, warp_21_conv4)

        return sythsis_output,warp_21_conv1,HR2_conv1
