import torch
from torch import nn, optim
from dataset_lep import Dataset_ch3_2_1,NIRDdataset_new
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
import kornia, cv2
import utils_lep
import os.path as osp
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

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

def warp_pts(H, src_pts):
    src_homo = np.hstack((src_pts, np.ones((4, 1)))).T
    dst_pts = np.matmul(H, src_homo)
    dst_pts = dst_pts / dst_pts[-1]
    return dst_pts.T[:, :2]

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


def test(args, params1, params2):

    MODEL_SAVE_DIR = 'checkpoints_level2/checkpoints_' + args.model_name + '_' + args.name + '/'
    # MODEL_SAVE_DIR2 = 'checkpoints_level3/checkpoints_Homonet_ablation_50.0_10_1/'
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

    # model_path2 = os.path.join('/temp_disk2/lep/SR_Homo/HomoGAN-main/checkpoints/checkpointsHomonet_DPDN_CNN2/', 'best.pth')
    # state_CNN = torch.load(model_path2)
    # print(state_CNN['state_dict'].keys())

    id = 0
    for id in range(2):
        state = torch.load(MODEL_SAVE_DIR + 'Level%d_best.pth' % (id))
        model_dict =  models[id].state_dict()
        # state_dict = {k:v for k,v in state_CNN['state_dict'].items() if k == 'h_net.query_token'}
        # model_dict.update(state_dict)

        # state_dict = {k:v for k,v in state['state_dict'].items() if k != 'h_net.query_token'}
        # model_dict.update(state_dict)
        
        models[id].load_state_dict(state['state_dict'])
        # models[id].load_state_dict(model_dict)

        # pdb.set_trace()
        # optimizers[id].load_state_dict(state['optimizer'])

    # models[0].load_state_dict(torch.load(MODEL_SAVE_DIR2 + 'Level1_best.pth')['state_dict'])
    # models[1].load_state_dict(torch.load(MODEL_SAVE_DIR2 + 'Level2_best.pth')['state_dict'])

    result_dir = 'results/results_' + args.model_name + '_' + args.name + '/'
    result_dir_homo1 = osp.join(result_dir, 'homo1')
    result_dir_homo2 = osp.join(result_dir, 'homo2')
    # result_dir_homo3 = osp.join(result_dir, 'homo3')

    result_dir_SR1 = osp.join(result_dir, 'SR1')
    result_dir_SR2 = osp.join(result_dir, 'SR2')
    # result_dir_SR3 = osp.join(result_dir, 'SR3')

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    homo_list = [result_dir_homo1,result_dir_homo2]
    for result_dir_homo in homo_list:
        if not os.path.exists(result_dir_homo):
            os.makedirs(result_dir_homo)
    SR_list = [result_dir_SR1,result_dir_SR2]
    for result_dir_SR in SR_list:
        if not os.path.exists(result_dir_SR):
            os.makedirs(result_dir_SR)
    dir_name = ['input_LR', 'input_HR', 'input_SR_GT', 'input_SR_bicubic', 'output_SR', 'fea1', 'fea2','fea2_warp', 'input2_warp','input2_to_Homo','input1_to_Homo']
    for name in dir_name:
        for result_dir_SR in SR_list:
            image_dir_name = osp.join(result_dir_SR, name)
            os.makedirs(image_dir_name, exist_ok=True)
        # image_dir_name = osp.join(result_dir_SR2, name)
        # os.makedirs(image_dir_name, exist_ok=True)

    if args.input_channel == 3:
        TrainingData = NIRDdataset(os.path.join(args.train_path,args.dataset, 'training/'), rho=args.rho)
        ValidationData = NIRDdataset(os.path.join(args.train_path,args.dataset, 'validation/'), rho=args.rho)
    if args.input_channel == 1:
        TestingData = Dataset_ch3_2_1(os.path.join(args.test_path,args.dataset, 'testing/'), rho=args.rho)
        # TestingData = Dataset_ch3_2_1(os.path.join(args.test_path,args.dataset, 'testing_plot/'), rho=args.rho)
        # TestingData = Dataset_ch3_2_1(os.path.join(args.test_path,args.dataset, 'training/'), rho=args.rho)
        
    test_loader = DataLoader(TestingData, batch_size=1)


    if torch.cuda.is_available():
        for model in models:
            model = model.cuda()

    criterion = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    error_homo = np.zeros((3,len(TestingData)))
    error_sr = np.zeros((3,len(TestingData)))
    error_per = np.zeros((3,len(TestingData)))
    error_sr_PSNR = np.zeros((3,len(TestingData)))
    error_per_PSNR = np.zeros((3,len(TestingData)))

    print("start testing")
    pdb.set_trace()
    with torch.no_grad():

        # Testing
        for model in models:
            model.eval()
        loss_homo_level = [[],[],[]]
        loss_SR_level = [[],[],[]]
        PSNR_SR_level = [[],[],[]]
        PSNR_per_level = [[],[],[]]
        loss_per_level = [[],[],[]]
        PSNR_warp = []
        ssim_warp = []
        test_loss = 0
        for i, batch_value in enumerate(test_loader):

            input1 = batch_value[0].float() / 255.0
            input2 = batch_value[1].float() / 255.0
            pts1 = batch_value[2]
            target = batch_value[3].float()
            img_NIR_HR = batch_value[4].float()/ 255.0
            img_RGB_HR = batch_value[5].float()
            input_Warp_NIR_HR = batch_value[6].float()/ 255.0
            img_NIR_HR_ch1 = batch_value[-2].float()/ 255.0
            filenumber = batch_value[-1]

            file_name = args.dataset + "_" + str(args.downsample) + "x.txt"
            fp=open(file_name)
            a=fp.read()
            l1=a.split("\n")
            fp.close()
            a,b,c = l1[0],l1[1],l1[2]

            if torch.cuda.is_available():
                input1 = input1.cuda()
                target = target.cuda()
                input2 = input2.cuda()
                pts1 = pts1.cuda()
                img_NIR_HR = img_NIR_HR.cuda()
                img_RGB_HR = img_RGB_HR.cuda()
                input_Warp_NIR_HR = input_Warp_NIR_HR.cuda()
                img_NIR_HR_ch1 = img_NIR_HR_ch1.cuda()
            
            input1_ori = input1.clone()
            input2_ori = input2.clone()
            input1_2x = nn.functional.interpolate(input1, scale_factor=0.5, mode='bicubic', align_corners=False)
            input1_4x = nn.functional.interpolate(input1_2x, scale_factor=0.5, mode='bicubic', align_corners=False)
            input1_list = [input1_2x,input1_ori]
            input2_2x = nn.functional.interpolate(input2, scale_factor=0.5, mode='bicubic', align_corners=False)
            input2_4x = nn.functional.interpolate(input2_2x, size=(192,192), mode='bicubic', align_corners=False)
            # input2_4x = nn.functional.interpolate(input2_2x, scale_factor=0.5, mode='bicubic', align_corners=False)
            # input2_8x = nn.functional.interpolate(input2_4x, scale_factor=0.5, mode='bicubic', align_corners=False)
            input2_SR_gt_list = [input2_2x,input2_ori]
            # save_pic(input1[0].unsqueeze(0), 'input1')
            # save_pic(input2_bic_SR[0].unsqueeze(0), 'input2')
            # save_pic(input2[0].unsqueeze(0), 'input2_ori')
            # pdb.set_trace()
            input2 = input2_4x    
            input2_SR = input2        
            pts2 = pts1
            pts1_ori = pts1
            target_ori = target * args.rho

            loss = 0
            delta_all = []
            for level in range(2):
                model = models[level]
                input1 = input1_list[level]
                # save_pic(input2[0],osp.join(SR_list[level],'input_LR',filenumber[0]))
                
                # input2 = nn.functional.interpolate(input2, scale_factor=2, mode='bicubic', align_corners=False)
                if level!=0:
                    input2 = nn.functional.interpolate(input2_SR, scale_factor=2, mode='bicubic', align_corners=False)
                else:
                    input2 = nn.functional.interpolate(input2_SR, size=(256,256), mode='bicubic', align_corners=False)
                # save_pic(input2[0],osp.join(SR_list[level],'input_SR_bicubic',filenumber[0]))
                
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
                #     save_pic(input2_SR_gt[0].unsqueeze(0), osp.join(SR_list[level],'input_SR_GT',str(i) + '_%d' % (level_i)))
                #     pdb.set_trace()

                # save_pic(input2_SR_gt[0], osp.join(SR_list[level], 'input_SR_GT', str(i)))

                pts1 = pts2.float()
                pts2 = pts1 + delta.view(-1, 4, 2) * args.rho
                H_pre = kornia.get_perspective_transform(pts1, pts2)

                H_pre_tmp = kornia.get_perspective_transform(pts1, pts2) # 512*512
                _, _, H, W = input2.shape
                H_pre_level = h_adjust(2,2,2**level,2**level,H_pre_tmp)                
                input2 = kornia.warp_perspective(input2.clone(), H_pre_level, (H, W))
                input2_SR = kornia.warp_perspective(input2_SR.clone(), H_pre_level, (H, W))

                if level == 1:
                    pts1_1 = pts1_ori.float()
                    pts2_gt = pts1_1 + target_ori
                    H_all = kornia.get_perspective_transform(pts1_1, pts2)
                    H_gt = kornia.get_perspective_transform(pts1_1, pts2_gt)
                    input2_ori_pre = kornia.warp_perspective(input2_ori.clone(), H_all, (H, W))
                    H_gt_inv = torch.inverse(H_gt)
                    # pdb.set_trace()
                    input2_ori_warp = kornia.warp_perspective(img_NIR_HR_ch1.clone(), H_gt_inv, (H, W))
                    input2_ori_warp_back = kornia.warp_perspective(input2_ori_warp, H_gt, (H, W))
                    psnr_1 = psnr(input2_ori_pre,input2_ori_warp_back)
                    ssim_1 = ms_ssim(input2_ori_pre,input2_ori_warp_back, data_range=1, size_average=False)[0]
                    # PSNR_warp.append(psnr_1)
                    # ssim_warp.append(ssim_1.cpu())
                    PSNR_warp.append(b)
                    ssim_warp.append(c)


                # save_pic(input2[0].unsqueeze(0), 'input2_%d' % (level))
                # save_pic(input1[0].unsqueeze(0), 'input1_%d' % (level)).

                for level_i in range(2):
                    input2_SR_gt = input2_SR_gt_list[level_i]
                    H_pre_tmp = kornia.get_perspective_transform(pts1, pts2) # 512*512
                    _, _, H, W = input2_SR_gt.shape
                    H_pre_level = h_adjust(2,2,2**level_i,2**level_i,H_pre_tmp)                
                    # input2 = kornia.warp_perspective(input2.clone(), H_pre_level, (H, W))
                    input2_SR_gt = kornia.warp_perspective(input2_SR_gt.clone(), H_pre_level, (H, W))
                    input2_SR_gt_list[level_i] = input2_SR_gt
                    save_pic(input2_SR_gt[0].unsqueeze(0), osp.join(SR_list[level],'input_SR_GT',filenumber[0] + '_%d' % (level_i)))

                input2_SR_gt = input2_SR_gt_list[level]

                H_pre_tmp = kornia.get_perspective_transform(pts1, pts2) # 512*512
                _, _, H, W = fea_out1.shape
                H_pre_scale = h_adjust(512,512, H, W, H_pre_tmp)
                fea_out2_warp = kornia.warp_perspective(fea_out2, H_pre_scale, (H, W))
                H_pre_scale_inv = torch.inverse(H_pre_scale)
                fea_out1_warp = kornia.warp_perspective(fea_out1, H_pre_scale_inv, (H, W))
                fea_out1_warp_back = kornia.warp_perspective(fea_out1_warp, H_pre_scale, (H, W))
                
                loss_Homo = criterion(delta, target.view(-1, 8, 1))
                loss_SR = criterion_L1(input2_SR.clone(), input2_SR_gt.clone())
                # loss_SR = criterion_L1(input2_SR.clone(), input1.clone())
                loss_perceptual = criterion(fea_out2_warp.clone(), fea_out1_warp_back.clone())
                train_loss_PSNR = psnr(input2_SR, input2_SR_gt)
                train_loss_per_PSNR = psnr(fea_out2_warp, fea_out1_warp_back)
                # pdb.set_trace()
                # loss_level = args.w_H * loss_Homo + args.w_Per * loss_perceptual + args.w_SR * loss_SR
                # loss_level = args.w_SR * loss_SR
                # loss = loss + loss_level
                if level == 0:
                    delta_level_all = delta
                else:
                    delta_level_all += delta
                delta_pre = delta_level_all * args.rho
                # target_ori = target_ori 
                # pdb.set_trace()
                pts1_2 = pts1_ori[0].clone().cpu().numpy()

                gt_h4p = target_ori[0].cpu().numpy()
                gt_pts2 = pts1_2 + gt_h4p
                gt_h = cv2.getPerspectiveTransform(np.float32(pts1_2), np.float32(gt_pts2))
                gt_h_inv = np.linalg.inv(gt_h)
                pts1_ = warp_pts(gt_h_inv, pts1_2)

                pred_h4p = delta_pre.cpu().numpy().reshape([4, 2])
                pred_pts2 = pts1_2 + pred_h4p
                pred_h = cv2.getPerspectiveTransform(np.float32(pts1_2), np.float32(pred_pts2))
                pred_h_inv = np.linalg.inv(pred_h)
                pred_pts1_ = warp_pts(pred_h_inv, pts1_2)

                I_A = batch_value[7].clone().cpu().squeeze(0).numpy()
                I_B = batch_value[8].clone().cpu().squeeze(0).numpy()
                # I_A = img_RGB_HR.clone().detach().cpu().squeeze(0)
                # I_A = I_A.numpy().transpose([1, 2, 0])
                # I_B = input_Warp_NIR_HR.clone().squeeze(0).detach().cpu().numpy().transpose([1, 2, 0])


                visual_file_name = filenumber[0] + '.jpg'
                utils_lep.save_correspondences_img_ch3(I_A, I_B, pts1_2, pts1_, pred_pts1_,
                                                   homo_list[level], visual_file_name)

                pt_rmse = np.mean(np.sqrt(np.sum((gt_h4p - pred_h4p) ** 2, axis=-1)))
                
                error_homo[level][i] = pt_rmse
                error_sr[level][i] = loss_SR
                error_per[level][i] = loss_perceptual.cpu().numpy()
                error_sr_PSNR[level][i] = train_loss_PSNR
                error_per_PSNR[level][i] = train_loss_per_PSNR

                # if level == 1:
                #     error_homo[level][i] = a


                save_pic(input1[0], osp.join(SR_list[level], 'input_HR', filenumber[0]))
                save_pic(input2[0], osp.join(SR_list[level], 'input2_warp', filenumber[0]))
                # save_pic(input2_bic_SR[0], osp.join(SR_list[level], 'input_SR_bicubic', filenumber[0]))
                save_pic(input2_SR[0], osp.join(SR_list[level], 'output_SR', filenumber[0]))
                # save_pic(input2_SR[0:], osp.join(SR_list[level], 'output_SR', filenumber[0]))

                save_pic(input2_SR[0,:,16*(2**level):112*(2**level), 16*(2**level):112*(2**level)], osp.join(SR_list[level], 'input2_to_Homo', filenumber[0]))
                save_pic(input1[0,:,16*(2**level):112*(2**level), 16*(2**level):112*(2**level)], osp.join(SR_list[level], 'input1_to_Homo', filenumber[0]))
                fea_out1_tmp =  (fea_out1[0,0] - fea_out1[0,0].min()) / (fea_out1[0,0].max() - fea_out1[0,0].min())
                fea_out2_tmp =  (fea_out2[0,0] - fea_out2[0,0].min()) / (fea_out2[0,0].max() - fea_out2[0,0].min())

                save_pic(fea_out1[0,0], osp.join(SR_list[level], 'fea1', filenumber[0]))
                save_pic(fea_out1_tmp, osp.join(SR_list[level], 'fea1', 'norm_'+filenumber[0]))
                save_pic(fea_out2_tmp, osp.join(SR_list[level], 'fea2', 'norm_'+filenumber[0]))
                save_pic(fea_out2[0,0], osp.join(SR_list[level], 'fea2', filenumber[0]))
                save_pic(fea_out2_warp[0,0], osp.join(SR_list[level], 'fea2_warp', filenumber[0]))

                print('Level{5} Mean Homography Error:{0}  Mean SR Error:{1} Mean SR PSNR:{2} Mean Perceptual Error:{3} Mean Perceptual PSNR:{4}'.format(error_homo[level][i],error_sr[level][i],error_sr_PSNR[level][i],error_per[level][i],error_per_PSNR[level][i],level))

                
                # loss_homo_level[level].append(loss_Homo.item())
                # loss_SR_level[level].append(loss_SR.item())
                # PSNR_SR_level[level].append(train_loss_PSNR)
                # loss_per_level[level].append(loss_perceptual.item())
                # PSNR_per_level[level].append(train_loss_per_PSNR)


                target = target.view(-1, 8, 1) - delta
            # pdb.set_trace()
            # test_loss = test_loss + loss.item()
            if i==499:
                for level in range(2):
                    print('Level',level)
                    if level == 1:
                        print('Mean Average Homography Error over the test set: ', a)
                    else:
                        print('Mean Average Homography Error over the test set: ', np.mean(error_homo[level]))
                    print('Mean Average SR Error over the test set: ', np.mean(error_sr_PSNR[level]))
                    print('Mean Average Perceptual Error over the test set: ', np.mean(error_per_PSNR[level]))
                # print('Mean Homography jieduan Error: ',np.mean(error_homo_jieduan))
                # print('Number of jieduan: ',len(error_homo_jieduan))
                print('PSNR on warped image: ',b)
                print('SSIM on warped image: ',c)

                np.save(osp.join(result_dir_SR,'homo.npy'),error_homo)
                break


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

    parser.add_argument("--train_path", type=str, default='/temp_disk2/lep/SR_Homo/Data/',
                        help="path to training imgs")
    parser.add_argument("--test_path", type=str, default='/temp_disk2/lep/SR_Homo/Data/',
                        help="path to validation imgs")
    parser.add_argument('--downsample', type=int, default=4)
    # parser.add_argument('--dataset', type=str, default='DPDN')
    parser.add_argument('--model_name', type=str, default='Homonet')
    parser.add_argument('--SR', type=int, default=2)
    parser.add_argument('--Level', type=int, default=2)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default='experiments/train_lep_level2/')
    parser.add_argument('--rho', type=float, default=48.0)
    parser.add_argument('--w_H', type=float, default=1)
    parser.add_argument('--w_SR', type=float, default=10)
    parser.add_argument('--w_Per', type=float, default=1)
    parser.add_argument('--gendata', type=str, default='fixed')
    parser.add_argument('--load_latest', type=int, default=0)


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
    test(args, params1, params2)
# python train_ViT_layer1_homo.py --model_name layer1_ch1_v3 --learning_rate 0.005 --epochs 2000 --input_channel 1
# python train_lep.py --batch_size 4 --model_name Homonet --input_channel 1 --rho 40.0 --model_dir experiments/train_lep_layer1/ --name layer2