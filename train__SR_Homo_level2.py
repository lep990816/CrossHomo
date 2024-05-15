import torch
from torch import nn, optim
from dataset_lep import  Dataset_ch3_2_1,NIRDdataset_new
# from model_vision_transformer.model_ViT_layer1_homo import VisionTransformer
# from model_vision_transformer.model_ViT_layer1_ori import VisionTransformer_v2
import model.net_cross_level as net
# from model.model_ViT_ori import VisionTransformer_ori
from torch.utils.data import DataLoader
import argparse
import time, random
import os, pdb,math
import os.path as osp
from torchvision import transforms
import numpy as np
from common import utils
from common.manager import Manager
import kornia

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def psnr(img1, img2):
    img1[img1 > 1.0] = 1.0
    img1[img1 < 0.0] = 0.0
    img2[img2 > 1.0] = 1.0
    img2[img2 < 0.0] = 0.0
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    # mse = np.mean((img1/255. - img2/255.) ** 2)
    mse = np.mean((img1 - img2) ** 2)

    pixel_max = 1.
    if math.sqrt(mse) != 0:
        psnr = 20 * math.log10(pixel_max / math.sqrt(mse))
    else:
        psnr = 20 * math.log10(pixel_max / (math.sqrt(mse) + 0.000001))
    return psnr

def save_pic(data, path):
    if osp.exists(path):
        os.system("rm " + path)
        print("rm " + path)
    reimage = data.cpu().clone()
    # reimage[reimage > 1.0] = 1.0
    # reimage[reimage < 0.0] = 0.0

    reimage = reimage.squeeze(0)
    # print(reimage.shape)
    reimage = transforms.ToPILImage()(reimage)  # PIL格式
    # print(reimage.size)
    # print(path)
    reimage.save(osp.join(path + '.png'))


best = 1000.0


def h_adjust(orishapea, orishapeb, resizeshapea, resizeshapeb, h_1):  # ->h_ori
    # a = original_img.shape[-2] / resized_img.shape[-2]
    # b = original_img.shape[-1] / resized_img.shape[-1]
    a = resizeshapea / orishapea
    b = resizeshapeb / orishapeb
    # pdb.set_trace()
    # the shape of H matrix should be (1, 3, 3)
    h_1[:, 0, :] = a * h_1[:, 0, :]
    h_1[:, :, 0] = (1. / a) * h_1[:, :, 0]
    h_1[:, 1, :] = b * h_1[:, 1, :]
    h_1[:, :, 1] = (1. / b) * h_1[:, :, 1]
    return h_1

freeze_list = ['fea_extra.0.weight', 'fea_extra.1.weight', 'fea_extra.1.bias', 'fea_extra.1.running_mean', 'fea_extra.1.running_var', 'fea_extra.1.num_batches_tracked', 
                'fea_extra.3.weight', 'fea_extra.4.weight', 'fea_extra.4.bias', 'fea_extra.4.running_mean', 'fea_extra.4.running_var', 'fea_extra.4.num_batches_tracked', 
                'fea_extra.6.weight', 'fea_extra.7.weight', 'fea_extra.7.bias', 'fea_extra.7.running_mean', 'fea_extra.7.running_var', 'fea_extra.7.num_batches_tracked',
                'h_net.feature_pyramid_extractor.convs.0.0.weight', 'h_net.feature_pyramid_extractor.convs.0.0.bias', 'h_net.feature_pyramid_extractor.convs.0.2.weight', 'h_net.feature_pyramid_extractor.convs.0.2.bias',
                'h_net.feature_pyramid_extractor.convs.1.0.weight', 'h_net.feature_pyramid_extractor.convs.1.0.bias', 'h_net.feature_pyramid_extractor.convs.1.2.weight', 'h_net.feature_pyramid_extractor.convs.1.2.bias']

def train(args, params1, params2):
    global best
    seed = random.randint(1, 1000)
    print("===> Random Seed: [%d]" % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    MODEL_SAVE_DIR = 'checkpoints_level2/checkpoints_' + args.model_name + '_' + args.name + '/'
    MODEL_SAVE_DIR2 = 'checkpoints_level3/checkpoints_Homonet_NIR_test4/'
    MODEL_SAVE_DIR2 = 'checkpoints_level3/checkpoints_Homonet_COCO_v2_10_10_1/'
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # if args.model_name == 'ori':
    #     model = VisionTransformer_ori.from_name('ViT-B_16')
    # if args.model_name == 'layer1_v1':
    #     model = VisionTransformer.from_name('ViT-B_16_lep_v1')
    # if args.model_name =='layer1_v2':
    #     model = VisionTransformer_v2.from_name('ViT-B_16_lep_v2')
    # if args.model_name =='layer1_v3':
    #     model = VisionTransformer_v3.from_name('ViT-B_16_lep_v3')
    # if args.model_name =='layer1_ch1_v3':
    #     model = VisionTransformer_v3.from_name('ViT-B_16_lep_v3_ch1')
    # if args.model_name =='layer1_v4':
    #     model = VisionTransformer_v3.from_name('ViT-B_16_lep_v3_layer12')
    if args.model_name == 'Homonet':
        model1 = net.fetch_net_lep(params1,1)
        model2 = net.fetch_net_lep(params2,2)
        # model3 = net.fetch_net_lep(params3,2)
    models = [model1, model2]

    # if args.input_channel == 3:
    #     TrainingData = NIRDdataset(os.path.join(args.train_path, 'training/'), rho=args.rho)
    #     ValidationData = NIRDdataset(os.path.join(args.train_path, 'validation/'), rho=args.rho)
    if args.input_channel == 1:
            
        if args.gendata == 'random':
            TrainingData = NIRDdataset_new(args.dataset, rho=args.rho)
        else:
            TrainingData = Dataset_ch3_2_1(os.path.join(args.train_path, 'training/'), rho=args.rho)
            ValidationData = Dataset_ch3_2_1(os.path.join(args.train_path, 'validation/'), rho=args.rho)
    print('Found totally {} training files and {} validation files'.format(len(TrainingData), len(ValidationData)))
    train_loader = DataLoader(TrainingData, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(ValidationData, batch_size=args.batch_size, num_workers=1)

    if torch.cuda.is_available():
        for model in models:
            model = model.cuda()

    criterion = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    if args.optim == 'sgd':
        optimizer1 = optim.SGD(models[0].parameters(), lr=args.learning_rate, momentum=0.9)
        optimizer2 = optim.SGD(models[1].parameters(), lr=args.learning_rate, momentum=0.9)
        # optimizer3 = optim.SGD(models[2].parameters(), lr=args.learning_rate, momentum=0.9)
    if args.optim == 'adam':
        optimizer1 = optim.Adam(models[0].parameters(), lr=params1.learning_rate / 4)
        optimizer2 = optim.Adam(models[1].parameters(), lr=params2.learning_rate / 2)
        # optimizer3 = optim.Adam(models[2].parameters(), lr=params3.learning_rate)
    optimizers = [optimizer1, optimizer2]

    # decrease the learning rate after every 1/3 epochs
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=int(args.epochs / 3), gamma=0.1)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=int(args.epochs / 3), gamma=0.1)
    # scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=int(args.epochs / 3), gamma=0.1)
    schedulers = [scheduler1, scheduler2]
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    # if os.path.exists(MODEL_SAVE_DIR + '/homographymodel' + '_iter_' + str(args.start_epochs) + '.pth'):
    #     print('===> Loading pre-trained model...')
    #     state = torch.load(MODEL_SAVE_DIR + '/homographymodel' + '_iter_' + str(args.start_epochs) + '.pth')
    #     model.load_state_dict(state['state_dict'])
    #     optimizer.load_state_dict(state['optimizer'])
    if args.load_latest == 1:
        print('===> Loading best model...')
        id = 0
        state_freeze = torch.load('/temp_disk2/lep/SR_Homo/HomoGAN-main/checkpoints/checkpointsHomonet_DPDN_CNN2/best.pth')
        for id in range(2):
            state = torch.load(MODEL_SAVE_DIR + 'Level%d_best.pth' % (id))
            # state = torch.load(MODEL_SAVE_DIR2 + 'Level%d_best.pth' % (id+1))
            model_dict =  models[id].state_dict()
            # state_dict = {k:v for k,v in state['state_dict'].items() if k not in freeze_list}
            # model_dict.update(state_dict)
            # state_dict = {k:v for k,v in state_freeze['state_dict'].items() if k in freeze_list}
            # model_dict.update(state_dict)
            models[id].load_state_dict(state['state_dict'])
            optimizers[id].load_state_dict(state['optimizer'])

    # for id in range(3):
    #     for name, para in models[id].named_parameters():
    #         if name in freeze_list:
    #             para.requires_grad_(False)

    print("start training")
    glob_iter = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Training
        for model in models:
            model.train()
        loss_homo_level = [[],[],[]]
        loss_SR_level = [[],[],[]]
        PSNR_SR_level = [[],[],[]]
        PSNR_per_level = [[],[],[]]
        loss_per_level = [[],[],[]]
        train_loss = 0
        for i, batch_value in enumerate(train_loader):
            # save model
            if (glob_iter % 100000 == 0 and glob_iter != 0):
                id = 0
                for model in models:
                    filename = 'Level%d_iter_%d.pth' % (id, glob_iter)
                    model_save_path = os.path.join(MODEL_SAVE_DIR, filename)
                    state = {'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizers[id].state_dict()}
                    torch.save(state, model_save_path)
                    id = id+1

            input1 = batch_value[0].float()/ 255.0
            input2 = batch_value[1].float()/ 255.0
            pts1 = batch_value[2]
            target = batch_value[3].float()

            if torch.cuda.is_available():
                input1 = input1.cuda() 
                target = target.cuda()
                input2 = input2.cuda()
                pts1_ori = pts1.cuda()
            
            input1_ori = input1.clone()
            input2_ori = input2.clone()
            input1_2x = nn.functional.interpolate(input1, scale_factor=0.5, mode='bicubic', align_corners=False)
            input1_4x = nn.functional.interpolate(input1_2x, scale_factor=0.5, mode='bicubic', align_corners=False)
            input1_list = [input1_2x,input1_ori]
            input2_2x = nn.functional.interpolate(input2, scale_factor=0.5, mode='bicubic', align_corners=False)
            input2_4x = nn.functional.interpolate(input2_2x, scale_factor=0.5, mode='bicubic', align_corners=False)
            input2_8x = nn.functional.interpolate(input2_4x, scale_factor=0.5, mode='bicubic', align_corners=False)
            input2_SR_gt_list = [input2_2x,input2_ori]
            # save_pic(input1[0].unsqueeze(0), 'input1')
            # save_pic(input2_bic_SR[0].unsqueeze(0), 'input2')
            # save_pic(input2[0].unsqueeze(0), 'input2_ori')
            # pdb.set_trace()
            input2 = input2_4x
            input2_SR = input2            
            pts2 = pts1_ori
            for optimizer in optimizers:
                optimizer.zero_grad() # 梯度清零
            loss = 0
            delta_all = []
            for level in range(2):
                model = models[level]
                input1 = input1_list[level]
                input2 = nn.functional.interpolate(input2_SR, scale_factor=2, mode='bicubic', align_corners=False)
                input2_SR_gt = input2_SR_gt_list[level]
                # pdb.set_trace()
                # input1 = input1[:,:,20*(2**level):116*(2**level), 20*(2**level):116*(2**level)]
                # input2 = input2[:,:,20*(2**level):116*(2**level), 20*(2**level):116*(2**level)]
                delta, fea_out1, fea_out2, input2_SR = model(torch.cat((input1, input2),dim=1))

                # delta_all.append(delta)
                # pts2_tmp = pts1_ori.clone()

                # for level_i in range(len(delta_all)):
                #     delta_tmp = delta_all[level_i]
                #     pts1_tmp = pts2_tmp.float()
                #     pts2_tmp = pts1_tmp + delta_tmp.view(-1, 4, 2) * args.rho
                #     H_pre_all = kornia.get_perspective_transform(pts1_tmp, pts2_tmp)
                #     _, _, H, W = input2_SR_gt.shape
                #     H_pre_level_all = h_adjust(4,4,2**level,2**level,H_pre_all)                
                #     input2_SR_gt = kornia.warp_perspective(input2_SR_gt, H_pre_level_all, (H, W))
                    # save_pic(input2_SR_gt[0].unsqueeze(0), 'input2_SR_gt_%d' % (i))
                    # pdb.set_trace()


                # if level == 0:
                #     delta_all = delta
                # else:
                #     delta_all = delta_all + delta

                # # input2_SR_gt = kornia.warp_perspective(input2_SR_gt, H_pre_level, (H, W))
                # pts1_tmp = pts1_ori.clone().float()
                # pts2_tmp = pts1_tmp + delta_all.view(-1, 4, 2) * args.rho
                # H_pre_all = kornia.get_perspective_transform(pts1_tmp, pts2_tmp)
                # _, _, H, W = input2_SR_gt.shape
                # H_pre_level_all = h_adjust(4,4,2**level,2**level,H_pre_all)                
                # input2_SR_gt = kornia.warp_perspective(input2_SR_gt, H_pre_level_all, (H, W))
                
                pts1 = pts2.float()
                pts2 = pts1 + delta.view(-1, 4, 2) * args.rho
                H_pre = kornia.get_perspective_transform(pts1, pts2)

                H_pre_tmp = kornia.get_perspective_transform(pts1, pts2) # 512*512
                _, _, H, W = input2.shape
                H_pre_level = h_adjust(2,2,2**level,2**level,H_pre_tmp)                
                input2 = kornia.warp_perspective(input2.clone(), H_pre_level, (H, W))
                input2_SR = kornia.warp_perspective(input2_SR.clone(), H_pre_level, (H, W))
                
                save_pic(input2[0].unsqueeze(0), 'input2_%d' % (level))
                save_pic(input1[0].unsqueeze(0), 'input1_%d' % (level))
                for level_i in range(2):
                    input2_SR_gt = input2_SR_gt_list[level_i]
                    H_pre_tmp = kornia.get_perspective_transform(pts1, pts2) # 512*512
                    _, _, H, W = input2_SR_gt.shape
                    H_pre_level = h_adjust(2,2,2**level_i,2**level_i,H_pre_tmp)                
                    # input2 = kornia.warp_perspective(input2.clone(), H_pre_level, (H, W))
                    input2_SR_gt = kornia.warp_perspective(input2_SR_gt.clone(), H_pre_level, (H, W))
                    input2_SR_gt_list[level_i] = input2_SR_gt
                    # save_pic(input2_SR_gt[0].unsqueeze(0), osp.join(SR_list[level],'input_SR_GT',str(i) + '_%d' % (level_i)))

                input2_SR_gt = input2_SR_gt_list[level]


                H_pre_tmp = kornia.get_perspective_transform(pts1, pts2) # 512*512
                _, _, H, W = fea_out1.shape
                H_pre_scale = h_adjust(512,512, H, W, H_pre_tmp)
                fea_out2_warp = kornia.warp_perspective(fea_out2.clone(), H_pre_scale, (H, W))
                H_pre_scale_inv = torch.inverse(H_pre_scale)
                fea_out1_warp = kornia.warp_perspective(fea_out1.clone(), H_pre_scale_inv, (H, W))
                fea_out1_warp_back = kornia.warp_perspective(fea_out1_warp.clone(), H_pre_scale, (H, W))
                
                loss_Homo = criterion(delta, target.view(-1, 8, 1))
                loss_SR = criterion_L1(input2_SR.clone(), input2_SR_gt.clone())
                # loss_perceptual = criterion(fea_out2_warp.clone(), fea_out1_warp_back.clone())
                loss_perceptual = criterion_L1(fea_out2_warp.clone(), fea_out1_warp_back.clone())

                # pdb.set_trace()
                loss_level = args.w_H * loss_Homo + args.w_Per * loss_perceptual + args.w_SR * loss_SR
                # loss_level = args.w_SR * loss_SR
                loss = loss + loss_level
                target = target.view(-1, 8, 1) - delta

                train_loss_PSNR = psnr(input2_SR, input2_SR_gt)
                train_loss_per_PSNR = psnr(fea_out2_warp, fea_out1_warp_back)
                
                loss_homo_level[level].append(loss_Homo.item())
                loss_SR_level[level].append(loss_SR.item())
                PSNR_SR_level[level].append(train_loss_PSNR)
                loss_per_level[level].append(loss_perceptual.item())
                PSNR_per_level[level].append(train_loss_per_PSNR)
            # pdb.set_trace()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            train_loss = train_loss + loss.item()

            if (i + 1) % 200 == 0 or (i + 1) == len(train_loader):
                print("Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}]".format(epoch + 1, args.epochs, i + 1, len(train_loader)))
                for level in range(2):
                    print("Level: {} Total loss: {:.4f} Homo loss: {:.4f} SR loss: {:.4f} SR PSNR:{:.2f} Perceptual loss: {:.4f} Perceptual PSNR:{:.2f} lr={:.6f}".format(
                        level+1, train_loss / len(loss_homo_level[level]), np.mean(loss_homo_level[level]),
                        np.mean(loss_SR_level[level]), np.mean(PSNR_SR_level[level]), np.mean(loss_per_level[level]), np.mean(PSNR_per_level[level]),
                        schedulers[level].get_lr()[0]))
                # pdb.set_trace()
                # print("target: \t predicted: ",target.view(-1, 8)[0],outputs[0])
                loss_homo_level = [[],[],[]]
                loss_SR_level = [[],[],[]]
                PSNR_SR_level = [[],[],[]]
                PSNR_per_level = [[],[],[]]
                loss_per_level = [[],[],[]]
                train_loss = 0.0

            glob_iter += 1
        for level in range(2):
            schedulers[level].step()

        # Validation
        with torch.no_grad():
            for model in models:
                model.eval()
            loss_homo_level = [[],[],[]]
            loss_SR_level = [[],[],[]]
            PSNR_SR_level = [[],[],[]]
            PSNR_per_level = [[],[],[]]
            loss_per_level = [[],[],[]]
            val_loss = 0.0


            for i, batch_value in enumerate(val_loader):
                input1 = batch_value[0].float()/ 255.0
                input2 = batch_value[1].float()/ 255.0
                pts1 = batch_value[2]
                target = batch_value[3].float()

                if torch.cuda.is_available():
                    input1 = input1.cuda() 
                    target = target.cuda()
                    input2 = input2.cuda() 
                    pts1_ori = pts1.cuda()
                
                input1_ori = input1.clone()
                input2_ori = input2.clone()
                input1_2x = nn.functional.interpolate(input1, scale_factor=0.5, mode='bicubic', align_corners=False)
                input1_4x = nn.functional.interpolate(input1_2x, scale_factor=0.5, mode='bicubic', align_corners=False)
                input1_list = [input1_2x,input1_ori]
                input2_2x = nn.functional.interpolate(input2, scale_factor=0.5, mode='bicubic', align_corners=False)
                input2_4x = nn.functional.interpolate(input2_2x, scale_factor=0.5, mode='bicubic', align_corners=False)
                input2_8x = nn.functional.interpolate(input2_4x, scale_factor=0.5, mode='bicubic', align_corners=False)
                input2_SR_gt_list = [input2_2x,input2_ori]           
                input2 = input2_4x   
                input2_SR = input2         
                loss = 0
                delta_all = []
                pts2 = pts1_ori
                
                for level in range(2):
                    model = models[level]
                    input1 = input1_list[level]
                    input2 = nn.functional.interpolate(input2_SR, scale_factor=2, mode='bicubic', align_corners=False)
                    input2_SR_gt = input2_SR_gt_list[level]
                    # input1 = input1[:,:,20*(2**level):116*(2**level), 20*(2**level):116*(2**level)]
                    # input2 = input2[:,:,20*(2**level):116*(2**level), 20*(2**level):116*(2**level)]
                    delta, fea_out1, fea_out2, input2_SR = model(torch.cat((input1, input2),dim=1))

                    # delta_all.append(delta)
                    # pts2_tmp = pts1_ori.clone()
                    # for level_i in range(len(delta_all)):
                    #     delta = delta_all[level_i]
                    #     pts1_tmp = pts2_tmp.float()
                    #     pts2_tmp = pts1_tmp + delta.view(-1, 4, 2) * args.rho
                    #     H_pre_all = kornia.get_perspective_transform(pts1_tmp, pts2_tmp)
                    #     _, _, H, W = input2_SR_gt.shape
                    #     H_pre_level_all = h_adjust(4,4,2**level,2**level,H_pre_all)                
                    #     input2_SR_gt = kornia.warp_perspective(input2_SR_gt, H_pre_level_all, (H, W))
                        
                    pts1 = pts2.float()
                    pts2 = pts1 + delta.view(-1, 4, 2) * args.rho
                    H_pre = kornia.get_perspective_transform(pts1, pts2)

                    H_pre_tmp = kornia.get_perspective_transform(pts1, pts2) # 512*512
                    _, _, H, W = input2.shape
                    H_pre_level = h_adjust(2,2,2**level,2**level,H_pre_tmp)                
                    input2 = kornia.warp_perspective(input2, H_pre_level, (H, W))
                    input2_SR = kornia.warp_perspective(input2_SR.clone(), H_pre_level, (H, W))

                    for level_i in range(2):
                        input2_SR_gt = input2_SR_gt_list[level_i]
                        H_pre_tmp = kornia.get_perspective_transform(pts1, pts2) # 512*512
                        _, _, H, W = input2_SR_gt.shape
                        H_pre_level = h_adjust(2,2,2**level_i,2**level_i,H_pre_tmp)                
                        # input2 = kornia.warp_perspective(input2.clone(), H_pre_level, (H, W))
                        input2_SR_gt = kornia.warp_perspective(input2_SR_gt.clone(), H_pre_level, (H, W))
                        input2_SR_gt_list[level_i] = input2_SR_gt
                        # save_pic(input2_SR_gt[0].unsqueeze(0), osp.join(SR_list[level],'input_SR_GT',str(i) + '_%d' % (level_i)))

                    input2_SR_gt = input2_SR_gt_list[level]

                    H_pre_tmp = kornia.get_perspective_transform(pts1, pts2) # 512*512
                    _, _, H, W = fea_out1.shape
                    H_pre_scale = h_adjust(512,512, H, W, H_pre_tmp)
                    fea_out2_warp = kornia.warp_perspective(fea_out2, H_pre_scale, (H, W))
                    H_pre_scale_inv = torch.inverse(H_pre_scale)
                    fea_out1_warp = kornia.warp_perspective(fea_out1, H_pre_scale_inv, (H, W))
                    fea_out1_warp_back = kornia.warp_perspective(fea_out1_warp, H_pre_scale, (H, W))
                    
                    loss_Homo = criterion(delta, target.view(-1, 8, 1))
                    loss_SR = criterion_L1(input2_SR, input2_SR_gt)
                    loss_perceptual = criterion_L1(fea_out2_warp, fea_out1_warp_back)

                    loss_level = args.w_H * loss_Homo + args.w_SR * loss_SR + args.w_Per * loss_perceptual
                    loss += loss_level
                    target = target.view(-1, 8, 1) - delta

                    train_loss_PSNR = psnr(input2_SR, input2_SR_gt)
                    train_loss_per_PSNR = psnr(fea_out2_warp, fea_out1_warp_back)
                    
                    loss_homo_level[level].append(loss_Homo.item())
                    loss_SR_level[level].append(loss_SR.item())
                    PSNR_SR_level[level].append(train_loss_PSNR)
                    loss_per_level[level].append(loss_perceptual.item())
                    PSNR_per_level[level].append(train_loss_per_PSNR)

                val_loss += loss.item()
            for level in range(2):
                print(
                "Validation: Epoch[{:0>3}/{:0>3}] Level:{} Total loss:{:.4f} Homo loss: {:.4f} SR loss: {:.4f} SR PSNR: {:.2f} Perceptual loss: {:.4f} Perceptual PSNR: {:.2f}, epoch time: {:.1f}s ".format(
                    epoch + 1, args.epochs, val_loss / len(val_loader), level, np.mean(loss_homo_level[level]),
                    np.mean(loss_SR_level[level]), np.mean(PSNR_SR_level[level]), np.mean(loss_per_level[level]), np.mean(PSNR_per_level[level]), time.time() - epoch_start))
            if np.mean(loss_homo_level[1]) < best:
                id = 0
                for model in models:
                    filename = 'Level%d_best.pth' % (id)
                    model_save_path = os.path.join(MODEL_SAVE_DIR, filename)
                    state = {'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizers[id].state_dict()}
                    torch.save(state, model_save_path)
                    id += 1
                
                best = np.mean(loss_homo_level[2])
            id = 0
            for model in models:
                filename = 'Level%d_lastest.pth' % (id)
                model_save_path = os.path.join(MODEL_SAVE_DIR, filename)
                state = {'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizers[id].state_dict()}
                torch.save(state, model_save_path)
                id += 1
                
    elapsed_time = time.time() - t0
    print("Finished Training in {:.0f}h {:.0f}m {:.0f}s.".format(
        elapsed_time // 3600, (elapsed_time % 3600) // 60, (elapsed_time % 3600) % 60))


if __name__ == "__main__":
    train_path = '/temp_disk2/lep/SR_Homo/Data/NIR_224/'
    val_path = '/temp_disk2/lep/SR_Homo/Data/NIR_224/'

    # train_path = 'data_crop/training_NIR/'
    # val_path = 'data_crop/validation_NIR/'

    total_iteration = 90000
    batch_size = 64
    num_samples = 10000
    steps_per_epoch = num_samples // batch_size
    epochs = int(total_iteration / steps_per_epoch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='DPDN_small', help="name")
    parser.add_argument("--dataset", type=str, default='NIR', help="name")

    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")

    parser.add_argument("--train_path", type=str, default='/temp_disk2/lep/SR_Homo/Data/COCO/',
                        help="path to training imgs")
    parser.add_argument("--val_path", type=str, default='/temp_disk2/lep/SR_Homo/Data/COCO/',
                        help="path to validation imgs")
    parser.add_argument('--downsample', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='Homonet')
    parser.add_argument('--SR', type=int, default=2)
    parser.add_argument('--Level', type=int, default=2)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default='experiments/train_lep_level2/')
    parser.add_argument('--rho', type=float, default=48.0)
    parser.add_argument('--w_H', type=float, default=10)
    parser.add_argument('--w_SR', type=float, default=1)
    parser.add_argument('--w_Per', type=float, default=1)
    parser.add_argument('--gendata', type=str, default='fixed')
    parser.add_argument('--load_latest', type=int, default=1)


    args = parser.parse_args()
    json_path1 = os.path.join(args.model_dir, 'params1.json')
    assert os.path.isfile(json_path1), "No json configuration file found at {}".format(json_path1)
    params1 = utils.Params(json_path1)
    json_path2 = os.path.join(args.model_dir, 'params2.json')
    assert os.path.isfile(json_path2), "No json configuration file found at {}".format(json_path2)
    params2 = utils.Params(json_path2)
    # json_path3 = os.path.join(args.model_dir, 'params3.json')
    # assert os.path.isfile(json_path3), "No json configuration file found at {}".format(json_path3)
    # params3 = utils.Params(json_path3)

    # Update args into params
    params1.update(vars(args))
    params2.update(vars(args))
    # params3.update(vars(args))
    train(args, params1, params2)
# python train_ViT_layer1_homo.py --model_name layer1_ch1_v3 --learning_rate 0.005 --epochs 2000 --input_channel 1
# python train_lep.py --batch_size 4 --model_name Homonet --input_channel 1 --rho 40.0 --model_dir experiments/train_lep_layer1/ --name layer2