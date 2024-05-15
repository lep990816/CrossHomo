"""Defines the neural network, losss function and metrics"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .utils import get_warp_flow, upsample2d_flow_as, get_grid
# from .swin_multi import SwinTransformer
from .Homonet_lep import SwinTransformer_cross
from timm.models.layers import trunc_normal_
from .module.aspp import ASPP
# from utils import get_warp_flow, upsample2d_flow_as, get_grid
# from swin_multi import SwinTransformer
# from timm.models.layers import trunc_normal_
# from module.aspp import ASPP
from torchvision import transforms
import os.path as osp
import torch.nn.functional as F
from .net_utils import *
from .FlowNet_model import FlowNet
from .Backward_warp_layer import Backward_warp
import pdb
import os.path as osp
__all__ = ['HomoNet', 'Discriminator']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def save_pic(data, path):
    if osp.exists(path):
        os.system("rm " + path)
        print("rm " + path)
    reimage = data.cpu().clone()
    reimage[reimage > 1.0] = 1.0
    reimage[reimage < 0.0] = 0.0

    reimage = reimage.squeeze(0)
    # print(reimage.shape)
    reimage = transforms.ToPILImage()(reimage)  # PIL格式
    # print(reimage.size)
    # print(path)
    reimage.save(osp.join(path + '.png'))

def tensor_erode(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)

    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded


def tensor_dilation(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)

    dilation, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    return dilation


def gen_basis(h, w, is_qr=True, is_scale=True):
    basis_nb = 8
    grid = get_grid(1, h, w).permute(0, 2, 3, 1)  # 1, w, h, (x, y, 1)
    flow = grid[:, :, :, :2] * 0

    names = globals()
    for i in range(1, basis_nb + 1):
        names['basis_' + str(i)] = flow.clone()

    basis_1[:, :, :, 0] += grid[:, :, :, 0]  # [1, w, h, (x, 0)]
    basis_2[:, :, :, 0] += grid[:, :, :, 1]  # [1, w, h, (y, 0)]
    basis_3[:, :, :, 0] += 1  # [1, w, h, (1, 0)]
    basis_4[:, :, :, 1] += grid[:, :, :, 0]  # [1, w, h, (0, x)]
    basis_5[:, :, :, 1] += grid[:, :, :, 1]  # [1, w, h, (0, y)]
    basis_6[:, :, :, 1] += 1  # [1, w, h, (0, 1)]
    basis_7[:, :, :, 0] += grid[:, :, :, 0] ** 2  # [1, w, h, (x^2, xy)]
    basis_7[:, :, :, 1] += grid[:, :, :, 0] * grid[:, :, :, 1]  # [1, w, h, (x^2, xy)]
    basis_8[:, :, :, 0] += grid[:, :, :, 0] * grid[:, :, :, 1]  # [1, w, h, (xy, y^2)]
    basis_8[:, :, :, 1] += grid[:, :, :, 1] ** 2  # [1, w, h, (xy, y^2)]

    flows = torch.cat([names['basis_' + str(i)] for i in range(1, basis_nb + 1)], dim=0)
    if is_qr:
        flows_ = flows.view(basis_nb, -1).permute(1, 0)  # N, h, w, c --> N, h*w*c --> h*w*c, N
        flow_q, _ = torch.qr(flows_)
        flow_q = flow_q.permute(1, 0).reshape(basis_nb, h, w, 2)
        flows = flow_q

    if is_scale:
        max_value = flows.abs().reshape(8, -1).max(1)[0].reshape(8, 1, 1, 1)
        flows = flows / max_value

    return flows.permute(0, 3, 1, 2)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, n_classes=1):
        super(Discriminator, self).__init__()
        self.cls_head = self.cls_net(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_last = nn.Conv2d(512, n_classes, kernel_size=1, padding=0, stride=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def cls_net(input_channels, kernel_size=3, padding=1):
        layers = []
        channels = [input_channels * 2, 32, 64, 128, 256, 512]
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding, stride=2, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cls_head(x)
        bs = len(x)
        x = self.conv_last(x)
        x = self.pool(x).view(bs, -1)
        return x


class HomoNet(nn.Module):
    # 224*224
    def __init__(self, params, backbone, init_mode="resnet", norm_layer=nn.LayerNorm):
        super(HomoNet, self).__init__()

        self.init_mode = init_mode
        self.params = params
        self.fea_extra = self.feature_extractor(self.params.in_channels, 1)
        self.h_net = backbone(params, norm_layer=norm_layer)
        self.basis = gen_basis(self.params.crop_size[0], self.params.crop_size[1]).unsqueeze(0).reshape(1, 8, -1)
        self.apply(self._init_weights)
        self.mask_pred = self.mask_predictor(32)

    def _init_weights(self, m):
        if "swin" in self.init_mode:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        elif "resnet" in self.init_mode:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def feature_extractor(input_channels, out_channles, kernel_size=3, padding=1):
        layers = []
        channels = [input_channels // 2, 4, 8, out_channles]
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def mask_predictor(input_channels, reduction=1):
        layers = []
        layers.append(nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3,
                                stride=1, padding=1, groups=2, bias=False))
        layers.append(ASPP(in_channels=input_channels * 2, out_channels=input_channels // 4, dilations=(1, 2, 5, 1)))
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=input_channels // reduction, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=input_channels // reduction, out_channels=1, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, data_batch):
        img1_full, img2_full = data_batch["imgs_gray_full"][:, :1, :, :], data_batch["imgs_gray_full"][:, 1:, :, :]
        img1_patch, img2_patch = data_batch["imgs_gray_patch"][:, :1, :, :], data_batch["imgs_gray_patch"][:, 1:, :, :]
        bs, _, h_patch, w_patch = data_batch["imgs_gray_patch"].size()
        start, src_pt = data_batch['start'], data_batch['pts']

        # ==========================full features======================================
        img1_patch_fea, img2_patch_fea = list(map(self.fea_extra, [img1_patch, img2_patch]))
        img1_full_fea, img2_full_fea = list(map(self.fea_extra, [img1_full, img2_full]))

        # ========================forward ====================================

        forward_fea = torch.cat([img1_patch_fea, img2_patch_fea], dim=1)
        weight_f = self.h_net(forward_fea)
        H_flow_f = (self.basis.to(forward_fea.device) * weight_f).sum(1).reshape(bs, 2, h_patch, w_patch)

        # ========================backward===================================
        backward_fea = torch.cat([img2_patch_fea, img1_patch_fea], dim=1)
        weight_b = self.h_net(backward_fea)
        H_flow_b = (self.basis.to(backward_fea.device) * weight_b).sum(1).reshape(bs, 2, h_patch, w_patch)

        if self.training:
            warp_img1_patch, warp_img1_patch_fea = list(
                map(lambda x: get_warp_flow(x, H_flow_b, start), [img1_full, img1_full_fea]))
            warp_img2_patch, warp_img2_patch_fea = list(
                map(lambda x: get_warp_flow(x, H_flow_f, start), [img2_full, img2_full_fea]))
        else:
            warp_img1_patch, warp_img1_patch_fea = list(
                map(lambda x: get_warp_flow(x, H_flow_b, start), [img1_patch, img1_patch_fea]))
            warp_img2_patch, warp_img2_patch_fea = list(
                map(lambda x: get_warp_flow(x, H_flow_f, start), [img2_patch, img2_patch_fea]))

        img1_patch_warp_fea, img2_patch_warp_fea = list(
            map(self.fea_extra, [warp_img1_patch, warp_img2_patch]))
        # ============================= mask==========================================
        if self.params.pretrain_phase:
            img1_patch_mask, img2_patch_mask, warp_img1_patch_mask, warp_img2_patch_mask = None, None, None, None
        else:
            if self.params.mask_use_fea:
                img1_patch_mask = self.mask_pred(
                    torch.cat((img1_patch_fea.detach(), warp_img2_patch_fea.detach()), dim=1))
                img2_patch_mask = self.mask_pred(
                    torch.cat((img2_patch_fea.detach(), warp_img1_patch_fea.detach()), dim=1))

            else:
                img1_patch_mask = self.mask_pred(torch.cat((img1_patch, warp_img2_patch), dim=1))
                img2_patch_mask = self.mask_pred(torch.cat((img2_patch, warp_img1_patch), dim=1))

            warp_img1_patch_mask = get_warp_flow(img1_patch_mask, H_flow_b, start)
            warp_img2_patch_mask = get_warp_flow(img2_patch_mask, H_flow_f, start)

        if not self.training:
            H_flow_f = upsample2d_flow_as(H_flow_f, img1_full, mode="bilinear", if_rate=True)
            H_flow_b = upsample2d_flow_as(H_flow_b, img1_full, mode="bilinear", if_rate=True)
        H_flow_f, H_flow_b = H_flow_f.permute(0, 2, 3, 1), H_flow_b.permute(0, 2, 3, 1)

        return {'warp_img1_patch_fea': warp_img1_patch_fea, 'warp_img2_patch_fea': warp_img2_patch_fea,
                'img1_patch_warp_fea': img1_patch_warp_fea, 'img2_patch_warp_fea': img2_patch_warp_fea,
                'warp_img1_patch': warp_img1_patch, 'warp_img2_patch': warp_img2_patch,
                'img1_patch_fea': img1_patch_fea, 'img2_patch_fea': img2_patch_fea,
                'flow_f': H_flow_f, 'flow_b': H_flow_b,
                'img1_patch_mask': img1_patch_mask, 'img2_patch_mask': img2_patch_mask,
                'warp_img1_patch_mask': warp_img1_patch_mask, 'warp_img2_patch_mask': warp_img2_patch_mask
                }

class MultiscaleWarpingNet(nn.Module):

    def __init__(self):
        super(MultiscaleWarpingNet, self).__init__()
        self.FlowNet = FlowNet(2)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(1)
        self.UNet_decoder_2 = UNet_decoder_2()


    def forward(self, input1_HR,input2_bic_SR):

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

        sythsis_output = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1,warp_21_conv2, warp_21_conv3,warp_21_conv4)

        return sythsis_output

class MultiscaleWarpingNet3(nn.Module):

    def __init__(self):
        super(MultiscaleWarpingNet3, self).__init__()
        self.FlowNet = FlowNet(2)
        self.Backward_warp = Backward_warp()
        self.layer_f1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size = 5, stride = 1, padding = 2),
                nn.ReLU(inplace = True))

        self.conv11 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 5, stride = 1, padding = 2),
                nn.ReLU(inplace = True))

        self.conv21 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(inplace = True))

        self.conv31 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(inplace = True))

        self.conv41 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(inplace = True))

        self.layer_f2 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size = 5, stride = 1, padding = 2),
                nn.ReLU(inplace = True))

        self.conv12 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 5, stride = 1, padding = 2),
                nn.ReLU(inplace = True))

        self.conv22 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(inplace = True))

        self.conv32 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(inplace = True))

        self.conv42 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 5, stride = 2, padding = 2),
                nn.ReLU(inplace = True))

        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, input1_HR, input2_bic_SR):
        # input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
        # input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
        # input_img1_SR = torch.from_numpy(buff['input_img1_SR']).cuda()

        # input_img2_LR = torch.from_numpy(buff['input_img2_LR']).cuda()
        # input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()
        # input_img2_SR = torch.from_numpy(buff['input_img2_SR']).cuda()

        flow = self.FlowNet(input2_bic_SR, input1_HR)
        flow_inv = self.FlowNet(input1_HR, input2_bic_SR)

        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']
        flow_21_1 = flow_inv['flow_12_1']
        flow_21_2 = flow_inv['flow_12_2']
        flow_21_3 = flow_inv['flow_12_3']
        flow_21_4 = flow_inv['flow_12_4']

        conv_HR = self.layer_f1(input1_HR)
        conv_LR = self.layer_f2(input2_bic_SR)
        conv1_HR = self.conv11(conv_HR)
        conv1_LR = self.conv12(conv_LR)
        conv1_HR_tmp = self.Backward_warp(conv1_HR, flow_12_1)
        conv1_LR_tmp = self.Backward_warp(conv1_LR, flow_21_1)

        b, c, h, w = conv1_HR_tmp.shape
        cat1 = torch.cat((conv1_HR_tmp, conv1_LR_tmp), dim=1)
        cat1_reshape = torch.reshape(cat1, (b, 2, c, h, w))
        cat1_transpose = cat1_reshape.transpose(2, 1)
        cat1_view = cat1_transpose.reshape(shape=[b, -1, h, w])
        conv1_HR = cat1_view[:, :c, :, :]
        conv1_LR = cat1_view[:, c:, :, :]

        conv2_HR = self.conv21(conv1_HR)
        conv2_LR = self.conv22(conv1_LR)
        conv2_HR = self.Backward_warp(conv2_HR, flow_12_2)
        conv2_LR = self.Backward_warp(conv2_LR, flow_21_2)

        b, c, h, w = conv2_HR.shape
        cat2 = torch.cat((conv2_HR, conv2_LR), dim=1)
        cat2_reshape = torch.reshape(cat2, (b, 2, c, h, w))
        cat2_transpose = cat2_reshape.transpose(2, 1)
        cat2_view = cat2_transpose.reshape(shape=[b, -1, h, w])
        conv2_HR = cat2_view[:, :c, :, :]
        conv2_LR = cat2_view[:, c:, :, :]

        conv3_HR = self.conv32(conv2_HR)
        conv3_LR = self.conv32(conv2_LR)
        conv3_HR = self.Backward_warp(conv3_HR, flow_12_3)
        conv3_LR = self.Backward_warp(conv3_LR, flow_21_3)

        b, c, h, w = conv3_HR.shape
        cat3 = torch.cat((conv3_HR, conv3_LR), dim=1)
        cat3_reshape = torch.reshape(cat3, (b, 2, c, h, w))
        cat3_transpose = cat3_reshape.transpose(2, 1)
        cat3_view = cat3_transpose.reshape(shape=[b, -1, h, w])
        conv3_HR = cat3_view[:, :c, :, :]
        conv3_LR = cat3_view[:, c:, :, :]

        conv4_HR = self.conv42(conv3_HR)
        conv4_LR = self.conv42(conv3_LR)
        conv4_HR = self.Backward_warp(conv4_HR, flow_12_4)
        conv4_LR = self.Backward_warp(conv4_LR, flow_21_4)

        b, c, h, w = conv4_HR.shape
        cat4 = torch.cat((conv4_HR, conv4_LR), dim=1)
        cat4_reshape = torch.reshape(cat4, (b, 2, c, h, w))
        cat4_transpose = cat4_reshape.transpose(2, 1)
        cat4_view = cat4_transpose.reshape(shape=[b, -1, h, w])
        conv4_HR = cat4_view[:, :c, :, :]
        conv4_LR = cat4_view[:, c:, :, :]

        # SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input2_bic_SR)
        # HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input1_HR)

        # warp_21_conv1 = self.Backward_warp(conv1_HR, flow_12_1)
        # warp_21_conv2 = self.Backward_warp(conv2_HR, flow_12_2)
        # warp_21_conv3 = self.Backward_warp(conv3_HR, flow_12_3)
        # warp_21_conv4 = self.Backward_warp(conv4_HR, flow_12_4)

        sythsis_output = self.UNet_decoder_2(conv1_LR, conv2_LR, conv3_LR, conv4_LR, conv1_HR, conv2_HR,
                                             conv3_HR, conv4_HR)

        return sythsis_output,conv1_HR_tmp,conv1_HR,conv1_LR

class HomoNet_lep(nn.Module):
    # 224*224
    def __init__(self, params ,level, backbone, init_mode="resnet", norm_layer=nn.LayerNorm):
        super(HomoNet_lep, self).__init__()

        self.init_mode = init_mode
        self.params = params
        self.level = level 
        self.SR = MultiscaleWarpingNet()
        self.SR3 = MultiscaleWarpingNet3()
        self.fea_extra = self.feature_extractor(self.params.in_channels, 1)
        self.h_net = backbone(params, norm_layer=norm_layer)
        self.basis = gen_basis(self.params.crop_size[0], self.params.crop_size[1]).unsqueeze(0).reshape(1, 8, -1)
        self.apply(self._init_weights)
        self.mask_pred = self.mask_predictor(32)

    def _init_weights(self, m):
        if "swin" in self.init_mode:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        elif "resnet" in self.init_mode:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def feature_extractor(input_channels, out_channles, kernel_size=3, padding=1):
        layers = []
        channels = [input_channels // 2, 4, 8, out_channles]
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def mask_predictor(input_channels, reduction=1):
        layers = []
        layers.append(nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3,
                                stride=1, padding=1, groups=2, bias=False))
        layers.append(ASPP(in_channels=input_channels * 2, out_channels=input_channels // 4, dilations=(1, 2, 5, 1)))
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=input_channels // reduction, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=input_channels // reduction, out_channels=1, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    # def forward(self, input1_HR,input2_bic_SR):
    def forward(self, inputs):
        input1_HR = inputs[:,0:1,]
        input2_bic_SR = inputs[:,1:2,]
        # pdb.set_trace()
        # img1_full, img2_full = data_batch["imgs_gray_full"][:, :1, :, :], data_batch["imgs_gray_full"][:, 1:, :, :]
        # img1_patch, img2_patch = data_batch["imgs_gray_patch"][:, :1, :, :], data_batch["imgs_gray_patch"][:, 1:, :, :]
        # bs, _, h_patch, w_patch = data_batch["imgs_gray_patch"].size()
        # start, src_pt = data_batch['start'], data_batch['pts']
        # # ==========================super resolution======================================
        if self.params.SR == 1:
            input2_SR_ori = self.SR(input1_HR,input2_bic_SR)
        elif self.params.SR == 0:
            input2_SR_ori = input2_bic_SR
        elif self.params.SR == 2:
            input2_SR_ori,conv1_HR_tmp,conv1_HR,conv1_LR = self.SR3(input1_HR,input2_bic_SR)
        # input1_HR_crop = input1_HR[:,:,16*(2**self.level):112*(2**self.level), 24*(2**self.level):120*(2**self.level)]
        # input2_SR_crop = input2_SR_ori[:,:,16*(2**self.level):112*(2**self.level), 24*(2**self.level):120*(2**self.level)]
        input1_HR_crop = input1_HR[:,:,16*(2**self.level):112*(2**self.level), 16*(2**self.level):112*(2**self.level)]
        input2_SR_crop = input2_SR_ori[:,:,16*(2**self.level):112*(2**self.level), 16*(2**self.level):112*(2**self.level)]
        if self.params.downsample == 2:
            if self.level == 1:
                input1_HR_crop = input1_HR[:,:,48:336,48:336]
                input2_SR_crop = input2_SR_ori[:,:,48:336,48:336]
            else:
                input1_HR_crop = input1_HR[:,:,16*(2**self.level):112*(2**self.level), 16*(2**self.level):112*(2**self.level)]
                input2_SR_crop = input2_SR_ori[:,:,16*(2**self.level):112*(2**self.level), 16*(2**self.level):112*(2**self.level)]
        # pdb.set_trace()
        # # ==========================full features======================================
        img1_patch_fea, img2_patch_fea = list(map(self.fea_extra, [input1_HR_crop, input2_SR_crop]))
        # img1_full_fea, img2_full_fea = list(map(self.fea_extra, [img1_full, img2_full]))
        # save_pic(img1_patch_fea[0],'fea1_CNN')
        # save_pic(img2_patch_fea[0],'fea2_CNN')
        # # ========================homography estimation ====================================
        # pdb.set_trace()
        forward_fea = torch.cat([img1_patch_fea, img2_patch_fea], dim=1)
        # pdb.set_trace()
        delta,fea_out1,fea_out2 = self.h_net(forward_fea)
        return delta,fea_out1[1],fea_out2[1],input2_SR_ori
        # H_flow_f = (self.basis.to(forward_fea.device) * weight_f).sum(1).reshape(bs, 2, h_patch, w_patch)

        # # ========================backward===================================
        # backward_fea = torch.cat([img2_patch_fea, img1_patch_fea], dim=1)
        # weight_b = self.h_net(backward_fea)
        # H_flow_b = (self.basis.to(backward_fea.device) * weight_b).sum(1).reshape(bs, 2, h_patch, w_patch)
        #
        # if self.training:
        #     warp_img1_patch, warp_img1_patch_fea = list(
        #         map(lambda x: get_warp_flow(x, H_flow_b, start), [img1_full, img1_full_fea]))
        #     warp_img2_patch, warp_img2_patch_fea = list(
        #         map(lambda x: get_warp_flow(x, H_flow_f, start), [img2_full, img2_full_fea]))
        # else:
        #     warp_img1_patch, warp_img1_patch_fea = list(
        #         map(lambda x: get_warp_flow(x, H_flow_b, start), [img1_patch, img1_patch_fea]))
        #     warp_img2_patch, warp_img2_patch_fea = list(
        #         map(lambda x: get_warp_flow(x, H_flow_f, start), [img2_patch, img2_patch_fea]))
        #
        # img1_patch_warp_fea, img2_patch_warp_fea = list(
        #     map(self.fea_extra, [warp_img1_patch, warp_img2_patch]))
        # # ============================= mask==========================================
        # if self.params.pretrain_phase:
        #     img1_patch_mask, img2_patch_mask, warp_img1_patch_mask, warp_img2_patch_mask = None, None, None, None
        # else:
        #     if self.params.mask_use_fea:
        #         img1_patch_mask = self.mask_pred(
        #             torch.cat((img1_patch_fea.detach(), warp_img2_patch_fea.detach()), dim=1))
        #         img2_patch_mask = self.mask_pred(
        #             torch.cat((img2_patch_fea.detach(), warp_img1_patch_fea.detach()), dim=1))
        #
        #     else:
        #         img1_patch_mask = self.mask_pred(torch.cat((img1_patch, warp_img2_patch), dim=1))
        #         img2_patch_mask = self.mask_pred(torch.cat((img2_patch, warp_img1_patch), dim=1))
        #
        #     warp_img1_patch_mask = get_warp_flow(img1_patch_mask, H_flow_b, start)
        #     warp_img2_patch_mask = get_warp_flow(img2_patch_mask, H_flow_f, start)
        #
        # if not self.training:
        #     H_flow_f = upsample2d_flow_as(H_flow_f, img1_full, mode="bilinear", if_rate=True)
        #     H_flow_b = upsample2d_flow_as(H_flow_b, img1_full, mode="bilinear", if_rate=True)
        # H_flow_f, H_flow_b = H_flow_f.permute(0, 2, 3, 1), H_flow_b.permute(0, 2, 3, 1)

        # return {'warp_img1_patch_fea': warp_img1_patch_fea, 'warp_img2_patch_fea': warp_img2_patch_fea,
        #         'img1_patch_warp_fea': img1_patch_warp_fea, 'img2_patch_warp_fea': img2_patch_warp_fea,
        #         'warp_img1_patch': warp_img1_patch, 'warp_img2_patch': warp_img2_patch,
        #         'img1_patch_fea': img1_patch_fea, 'img2_patch_fea': img2_patch_fea,
        #         'flow_f': H_flow_f, 'flow_b': H_flow_b,
        #         'img1_patch_mask': img1_patch_mask, 'img2_patch_mask': img2_patch_mask,
        #         'warp_img1_patch_mask': warp_img1_patch_mask, 'warp_img2_patch_mask': warp_img2_patch_mask
        #         }

def Ms_Transformer(pretrained=False, **kwargs):
    """Constructs a Multi-scale Transformer model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HomoNet(backbone=SwinTransformer, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def Ms_Transformer_lep(level,pretrained=False, **kwargs):
    """Constructs a Multi-scale Transformer model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HomoNet_lep(backbone=SwinTransformer_cross,level=level, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def fetch_net(params):
    if params.net_type == "HomoGAN":
        HNet = Ms_Transformer(params=params)
    else:
        raise NotImplementedError
    if params.pretrain_phase:
        return HNet
    else:
        DNet = Discriminator()
        return HNet, DNet

def fetch_net_lep(params,level):
    HNet = Ms_Transformer_lep(params=params,level=level)
    return HNet

# v = HomoNet()
# img1 = torch.randn(1,3,224,224)
# img2 = torch.randn(1,3,224,224)
# preds = v(img1)
#
# a=1