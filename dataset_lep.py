import os,pdb,cv2,glob
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from numpy.linalg import inv

def ImagePreProcessing(source_path,image_number, rho, patch_size, imsize, types):
    image_name1 = 'Depth'

    image_path_1 = os.path.join(source_path, 'modal2', 'Depth_' + str(image_number+1) + '.bmp')
    image_path_2 = os.path.join(source_path, 'modal1', 'RGB_' + str(image_number+1)  + '.png')
    # pdb.set_trace()
    img1 = cv2.imread(image_path_1)
    img1 = cv2.resize(img1, imsize)

    img2 = cv2.imread(image_path_2)
    img2 = cv2.resize(img2, imsize)
    # print(image_number)

    
    position_p = (random.randint(rho, imsize[0] - rho - patch_size), random.randint(rho, imsize[1] - rho - patch_size))

    tl_point = position_p
    tr_point = (patch_size + position_p[0], position_p[1])
    br_point = (patch_size + position_p[0], patch_size + position_p[1])
    bl_point = (position_p[0], patch_size + position_p[1])

    test_image = img1.copy()
    four_points = [tl_point, tr_point, br_point, bl_point]

    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)

    warped_image = cv2.warpPerspective(img1, H_inverse, imsize)

    training_image = np.dstack((img2, warped_image))
    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    datum = (training_image, np.array(four_points), H_four_points, img1)

    return datum
    
class NIRDdataset_new(Dataset):
    def __init__(self, dataset, rho=40):
        # lst = os.listdir(path)
        # self.data = [path + i for i in lst]
        self.rho = rho
        self.train_path = os.path.join('/temp_disk2/lep/dataset',dataset,'train/')
        self.train_path1 = os.path.join(self.train_path,'modal1')
        self.train_path2 = os.path.join(self.train_path,'modal2')
        self.patch_size = 384
        self.imsize = (544,544)
        self.modal1_list = sorted(glob.glob(os.path.join(self.train_path1,"*")))
        self.modal2_list = sorted(glob.glob(os.path.join(self.train_path2,"*")))
        # pdb.set_trace()
    def __getitem__(self, index):

            (ori_images, pts1, delta, img_NIR_HR) = ImagePreProcessing(self.train_path,index,self.rho, self.patch_size, self.imsize, types='train')
            # (ori_images, pts1, delta, ori_images_2x) = x
            # ori_images_2x = (ori_images_2x.astype(float) - 127.5) / 127.5
            # pdb.set_trace()

            input_HR_ori = ori_images[:, :, 0:3]
            input_HR = cv2.cvtColor(input_HR_ori,cv2.COLOR_RGB2GRAY)
            # print(input_HR.shape)
            input_HR = input_HR[np.newaxis, :]
            # input_HR = np.transpose(input_HR, [2, 0, 1])
            input_HR_crop = input_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            

            input_Warp_NIR_HR_ori = ori_images[:, :, 3:6]
            input_Warp_NIR_HR = cv2.cvtColor(input_Warp_NIR_HR_ori,cv2.COLOR_RGB2GRAY)
            input_Warp_NIR_HR = input_Warp_NIR_HR[np.newaxis, :]
            # input_Warp_NIR_HR = np.transpose(input_Warp_NIR_HR, [2, 0, 1])
            input_Warp_NIR_HR_crop = input_Warp_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            

            img_NIR_HR = np.transpose(img_NIR_HR, [2, 0, 1])

            # pdb.set_trace()

            delta = delta.astype(float) / self.rho
            input1 = torch.from_numpy(input_HR_crop)
            input2 = torch.from_numpy(input_Warp_NIR_HR_crop)

            pts1 = torch.from_numpy(pts1)
            # pts2 = torch.from_numpy(pts2)
            delta = torch.from_numpy(delta)

            img_NIR_HR = torch.from_numpy(img_NIR_HR)
            return input1, input2, pts1, delta, img_NIR_HR,input_HR, input_Warp_NIR_HR,input_HR_ori,input_Warp_NIR_HR_ori

    def __len__(self):
        return len(self.modal1_list)

class NIRDdataset_ch1(Dataset):
    def __init__(self, path, rho=16):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho

    def __getitem__(self, index):
        # ori_images, pts1, delta, ori_images_2x = np.load(self.data[index], allow_pickle=True)

        x = np.load(self.data[index], allow_pickle=True)
        if x is not None:
            (ori_images, pts1, delta, img_NIR_HR) = x
            filename = self.data[index]
            filenumber = filename.split('/')[-1][2:-4]
            # (ori_images, pts1, delta, ori_images_2x) = x
            # ori_images_2x = (ori_images_2x.astype(float) - 127.5) / 127.5
            # pdb.set_trace()

            input_HR = ori_images[:, :, 0:1]
            input_HR = np.transpose(input_HR, [2, 0, 1])
            input_HR_crop = input_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]

            input_Warp_NIR_HR = ori_images[:, :, 1:2]
            input_Warp_NIR_HR = np.transpose(input_Warp_NIR_HR, [2, 0, 1])
            input_Warp_NIR_HR_crop = input_Warp_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]

            # img_NIR_HR = np.transpose(img_NIR_HR, [2, 0, 1])

            # pdb.set_trace()

            delta = delta.astype(float) / self.rho
            input1 = torch.from_numpy(input_HR_crop)
            input2 = torch.from_numpy(input_Warp_NIR_HR_crop)

            pts1 = torch.from_numpy(pts1)
            delta = torch.from_numpy(delta)

            img_NIR_HR = torch.from_numpy(img_NIR_HR)
            return input1, input2, pts1, delta, img_NIR_HR,input_HR, input_Warp_NIR_HR

    def __len__(self):
        return len(self.data)

class Dataset_ch3(Dataset):
    def __init__(self, path, rho=48):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho

    def __getitem__(self, index):
        # ori_images, pts1, delta, ori_images_2x = np.load(self.data[index], allow_pickle=True)

        x = np.load(self.data[index], allow_pickle=True)
        if x is not None:
            (ori_images, pts1, delta, img_NIR_HR_ori) = x
            filename = self.data[index]
            filenumber = filename.split('/')[-1][2:-4]
            # (ori_images, pts1, delta, ori_images_2x) = x
            # ori_images_2x = (ori_images_2x.astype(float) - 127.5) / 127.5
            # pdb.set_trace()

            input_HR_ori = ori_images[:, :, 0:3]
            input_HR = cv2.cvtColor(input_HR_ori,cv2.COLOR_RGB2GRAY)
            # print(input_HR.shape)
            input_HR = input_HR[np.newaxis, :]
            

            input_Warp_NIR_HR_ori = ori_images[:, :, 3:6]
            input_Warp_NIR_HR_ch1 = cv2.cvtColor(input_Warp_NIR_HR_ori,cv2.COLOR_RGB2GRAY)
            input_Warp_NIR_HR_ch1 = input_Warp_NIR_HR_ch1[np.newaxis, :]
            input_Warp_NIR_HR = np.transpose(input_Warp_NIR_HR_ori, [2, 0, 1])
            # input_Warp_NIR_HR_crop = input_Warp_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            
            
            img_NIR_HR = np.transpose(img_NIR_HR_ori, [2, 0, 1])
            img_NIR_HR_ch1 = cv2.cvtColor(img_NIR_HR_ori,cv2.COLOR_RGB2GRAY)
            img_NIR_HR_ch1 = img_NIR_HR_ch1[np.newaxis, :]
            img_NIR_HR_ch1 = torch.from_numpy(img_NIR_HR_ch1)

            # pdb.set_trace()

            delta = delta.astype(float) / self.rho
            input1 = torch.from_numpy(input_HR)
            input2 = torch.from_numpy(input_Warp_NIR_HR)

            pts1 = torch.from_numpy(pts1)
            # pts2 = torch.from_numpy(pts2)
            delta = torch.from_numpy(delta)
            input2_ch1 = torch.from_numpy(input_Warp_NIR_HR_ch1)

            img_NIR_HR = torch.from_numpy(img_NIR_HR)
            return input1, input2, pts1, delta, img_NIR_HR,input_HR, input_Warp_NIR_HR_ch1,input_HR_ori,input_Warp_NIR_HR_ori,img_NIR_HR_ch1,filenumber,input2_ch1

    def __len__(self):
        return len(self.data)

class Dataset_ch3_2_1(Dataset):
    def __init__(self, path, rho=48):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho

    def __getitem__(self, index):
        # ori_images, pts1, delta, ori_images_2x = np.load(self.data[index], allow_pickle=True)

        x = np.load(self.data[index], allow_pickle=True)
        if x is not None:
            (ori_images, pts1, delta, img_NIR_HR_ori) = x
            filename = self.data[index]
            filenumber = filename.split('/')[-1][2:-4]
            # (ori_images, pts1, delta, ori_images_2x) = x
            # ori_images_2x = (ori_images_2x.astype(float) - 127.5) / 127.5
            # pdb.set_trace()

            input_HR_ori = ori_images[:, :, 0:3]
            input_HR = cv2.cvtColor(input_HR_ori,cv2.COLOR_RGB2GRAY)
            # print(input_HR.shape)
            input_HR = input_HR[np.newaxis, :]
            # input_HR = np.transpose(input_HR, [2, 0, 1])
            # input_HR_crop = input_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            

            input_Warp_NIR_HR_ori = ori_images[:, :, 3:6]
            input_Warp_NIR_HR = cv2.cvtColor(input_Warp_NIR_HR_ori,cv2.COLOR_RGB2GRAY)
            input_Warp_NIR_HR = input_Warp_NIR_HR[np.newaxis, :]
            # input_Warp_NIR_HR = np.transpose(input_Warp_NIR_HR, [2, 0, 1])
            # input_Warp_NIR_HR_crop = input_Warp_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            
            
            img_NIR_HR = np.transpose(img_NIR_HR_ori, [2, 0, 1])
            img_NIR_HR_ch1 = cv2.cvtColor(img_NIR_HR_ori,cv2.COLOR_RGB2GRAY)
            img_NIR_HR_ch1 = img_NIR_HR_ch1[np.newaxis, :]
            img_NIR_HR_ch1 = torch.from_numpy(img_NIR_HR_ch1)

            # pdb.set_trace()

            delta = delta.astype(float) / self.rho
            input1 = torch.from_numpy(input_HR)
            input2 = torch.from_numpy(input_Warp_NIR_HR)

            pts1 = torch.from_numpy(pts1)
            # pts2 = torch.from_numpy(pts2)
            delta = torch.from_numpy(delta)

            img_NIR_HR = torch.from_numpy(img_NIR_HR)
            return input1, input2, pts1, delta, img_NIR_HR,input_HR, input_Warp_NIR_HR,input_HR_ori,input_Warp_NIR_HR_ori,img_NIR_HR_ch1,filenumber

    def __len__(self):
        return len(self.data)

class NIRDdataset_ch3_2_1(Dataset):
    def __init__(self, path, rho=48):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho

    def __getitem__(self, index):
        # ori_images, pts1, delta, ori_images_2x = np.load(self.data[index], allow_pickle=True)

        x = np.load(self.data[index], allow_pickle=True)
        if x is not None:
            (ori_images, pts1, delta, img_NIR_HR) = x
            filename = self.data[index]
            filenumber = filename.split('/')[-1][2:-4]
            # (ori_images, pts1, delta, ori_images_2x) = x
            # ori_images_2x = (ori_images_2x.astype(float) - 127.5) / 127.5
            # pdb.set_trace()

            input_HR_ori = ori_images[:, :, 0:3]
            input_HR = cv2.cvtColor(input_HR_ori,cv2.COLOR_RGB2GRAY)
            # print(input_HR.shape)
            input_HR = input_HR[np.newaxis, :]
            # input_HR = np.transpose(input_HR, [2, 0, 1])
            input_HR_crop = input_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            

            input_Warp_NIR_HR_ori = ori_images[:, :, 3:6]
            input_Warp_NIR_HR = cv2.cvtColor(input_Warp_NIR_HR_ori,cv2.COLOR_RGB2GRAY)
            input_Warp_NIR_HR = input_Warp_NIR_HR[np.newaxis, :]
            # input_Warp_NIR_HR = np.transpose(input_Warp_NIR_HR, [2, 0, 1])
            input_Warp_NIR_HR_crop = input_Warp_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            
            img_NIR_HR = cv2.cvtColor(img_NIR_HR,cv2.COLOR_RGB2GRAY)
            img_NIR_HR = img_NIR_HR[np.newaxis, :]
            img_NIR_HR_crop = img_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]


            # pdb.set_trace()

            delta = delta.astype(float) / self.rho
            input1 = torch.from_numpy(input_HR_crop)
            input2 = torch.from_numpy(input_Warp_NIR_HR_crop)


            pts1 = torch.from_numpy(pts1)
            # pts2 = torch.from_numpy(pts2)
            delta = torch.from_numpy(delta)

            img_NIR_HR = torch.from_numpy(img_NIR_HR)
            img_NIR_HR_crop = torch.from_numpy(img_NIR_HR_crop)
            return input1, input2, pts1, delta, img_NIR_HR,input_HR, input_Warp_NIR_HR,input_HR_ori,input_Warp_NIR_HR_ori,img_NIR_HR_crop,filenumber

    def __len__(self):
        return len(self.data)

class NIRDdataset_ch3(Dataset):
    def __init__(self, path, rho=48):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho

    def __getitem__(self, index):
        # ori_images, pts1, delta, ori_images_2x = np.load(self.data[index], allow_pickle=True)

        x = np.load(self.data[index], allow_pickle=True)
        if x is not None:
            (ori_images, pts1, delta, img_NIR_HR) = x
            filename = self.data[index]
            filenumber = filename.split('/')[-1][2:-4]
            # (ori_images, pts1, delta, ori_images_2x) = x
            # ori_images_2x = (ori_images_2x.astype(float) - 127.5) / 127.5
            # pdb.set_trace()

            input_HR_ori = ori_images[:, :, 0:3]
            input_HR = cv2.cvtColor(input_HR_ori,cv2.COLOR_RGB2GRAY)
            # print(input_HR.shape)
            input_HR = input_HR[np.newaxis, :]
            # input_HR = np.transpose(input_HR, [2, 0, 1])
            input_HR_crop = input_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            

            input_Warp_NIR_HR_ori = ori_images[:, :, 3:6]
            input_Warp_NIR_HR_ch1 = cv2.cvtColor(input_Warp_NIR_HR_ori,cv2.COLOR_RGB2GRAY)
            input_Warp_NIR_HR_ch1 = input_Warp_NIR_HR_ch1[np.newaxis, :]
            input_Warp_NIR_HR = np.transpose(input_Warp_NIR_HR_ori, [2, 0, 1])
            input_Warp_NIR_HR_crop = input_Warp_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]

            img_NIR_HR = cv2.cvtColor(img_NIR_HR,cv2.COLOR_RGB2GRAY)
            img_NIR_HR = img_NIR_HR[np.newaxis, :]
            img_NIR_HR_crop = img_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]


            # pdb.set_trace()

            delta = delta.astype(float) / self.rho
            input1 = torch.from_numpy(input_HR_crop)
            input2 = torch.from_numpy(input_Warp_NIR_HR_crop)


            pts1 = torch.from_numpy(pts1)
            # pts2 = torch.from_numpy(pts2)
            delta = torch.from_numpy(delta)

            img_NIR_HR = torch.from_numpy(img_NIR_HR)
            img_NIR_HR_crop = torch.from_numpy(img_NIR_HR_crop)
            return input1, input2, pts1, delta, img_NIR_HR,input_HR, input_Warp_NIR_HR_ch1,input_HR_ori,input_Warp_NIR_HR_ori,img_NIR_HR_crop,filenumber

    def __len__(self):
        return len(self.data)

class NIRDdataset_ch1_SR(Dataset):
    def __init__(self, path, rho=48):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho

    def __getitem__(self, index):
        # ori_images, pts1, delta, ori_images_2x = np.load(self.data[index], allow_pickle=True)

        x = np.load(self.data[index], allow_pickle=True)
        if x is not None:
            (ori_images, pts1, delta,img_NIR_HR) = x
            # (ori_images, pts1, delta, ori_images_2x) = x
            # ori_images_2x = (ori_images_2x.astype(float) - 127.5) / 127.5
            # pdb.set_trace()

            input_HR_ori = ori_images[:, :, 0:3]
            input_HR = cv2.cvtColor(input_HR_ori,cv2.COLOR_RGB2GRAY)
            # print(input_HR.shape)
            input_HR = input_HR[np.newaxis, :]
            # input_HR = np.transpose(input_HR, [2, 0, 1])
            input_HR_crop = input_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            

            input_Warp_NIR_HR_ori = ori_images[:, :, 3:6]
            input_Warp_NIR_HR = cv2.cvtColor(input_Warp_NIR_HR_ori,cv2.COLOR_RGB2GRAY)
            input_Warp_NIR_HR = input_Warp_NIR_HR[np.newaxis, :]
            # input_Warp_NIR_HR = np.transpose(input_Warp_NIR_HR, [2, 0, 1])
            input_Warp_NIR_HR_crop = input_Warp_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            

            # img_NIR_HR = np.transpose(img_NIR_HR, [2, 0, 1])

            # pdb.set_trace()

            delta = delta.astype(float) / self.rho
            input1 = torch.from_numpy(input_HR_crop)
            input2 = torch.from_numpy(input_Warp_NIR_HR_crop)

            pts1 = torch.from_numpy(pts1)
            # pts2 = torch.from_numpy(pts2)
            delta = torch.from_numpy(delta)

            # img_NIR_HR = torch.from_numpy(img_NIR_HR)
            return input1, input2, pts1, delta, input_HR, input_Warp_NIR_HR,input_HR_ori,input_Warp_NIR_HR_ori

    def __len__(self):
        return len(self.data)

class Dataset_ch1_delta(Dataset):
    def __init__(self, path, rho=48, ratio = 1):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho 
        self.ratio = ratio

    def __getitem__(self, index, ratio):
        # ori_images, pts1, delta, ori_images_2x = np.load(self.data[index], allow_pickle=True)

        x = np.load(self.data[index], allow_pickle=True)
        if x is not None:
            (ori_images, pts1, delta, img_NIR_HR) = x
            # (ori_images, pts1, delta, ori_images_2x) = x
            # ori_images_2x = (ori_images_2x.astype(float) - 127.5) / 127.5
            # pdb.set_trace()
            perturbed_four_points = pts1 + delta * self.ratio
            H = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(perturbed_four_points))
            H_inverse = inv(H)
            warped_image = cv2.warpPerspective(img_NIR_HR, H_inverse, (544,544))
            warped_image = cv2.cvtColor(warped_image,cv2.COLOR_RGB2GRAY)
            warped_image = warped_image[np.newaxis, :]
            warped_image_crop = warped_image[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]

            input_HR_ori = ori_images[:, :, 0:3]
            input_HR = cv2.cvtColor(input_HR_ori,cv2.COLOR_RGB2GRAY)
            # print(input_HR.shape)
            input_HR = input_HR[np.newaxis, :]
            # input_HR = np.transpose(input_HR, [2, 0, 1])
            input_HR_crop = input_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            

            input_Warp_NIR_HR_ori = ori_images[:, :, 3:6]
            input_Warp_NIR_HR = cv2.cvtColor(input_Warp_NIR_HR_ori,cv2.COLOR_RGB2GRAY)
            input_Warp_NIR_HR = input_Warp_NIR_HR[np.newaxis, :]
            # input_Warp_NIR_HR = np.transpose(input_Warp_NIR_HR, [2, 0, 1])
            input_Warp_NIR_HR_crop = input_Warp_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]
            

            img_NIR_HR = np.transpose(img_NIR_HR, [2, 0, 1])

            # pdb.set_trace()

            delta = delta.astype(float) / self.rho
            input1 = torch.from_numpy(input_HR_crop)
            input2 = torch.from_numpy(input_Warp_NIR_HR_crop)
            input2 = torch.from_numpy(warped_image_crop)

            pts1 = torch.from_numpy(pts1)
            # pts2 = torch.from_numpy(pts2)
            delta = torch.from_numpy(delta)

            img_NIR_HR = torch.from_numpy(img_NIR_HR)
            return input1, input2, pts1, delta, img_NIR_HR,input_HR, input_Warp_NIR_HR,input_HR_ori,input_Warp_NIR_HR_ori

    def __len__(self):
        return len(self.data)


class NIRDdataset_384(Dataset):
    def __init__(self, path, rho=40.0):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho

    def __getitem__(self, index):
        # ori_images, pts1, delta, ori_images_2x = np.load(self.data[index], allow_pickle=True)

        x = np.load(self.data[index], allow_pickle=True)
        if x is not None:
            (ori_images, pts1, delta, img_NIR_HR) = x
            # (ori_images, pts1, delta, ori_images_2x) = x
            # ori_images_2x = (ori_images_2x.astype(float) - 127.5) / 127.5
            # pdb.set_trace()

            input_HR = ori_images[:, :, 0:3]
            input_HR = np.transpose(input_HR, [2, 0, 1])
            input_HR_crop = input_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]

            input_Warp_NIR_HR = ori_images[:, :, 3:6]
            input_Warp_NIR_HR = np.transpose(input_Warp_NIR_HR, [2, 0, 1])
            input_Warp_NIR_HR_crop = input_Warp_NIR_HR[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]

            img_NIR_HR = np.transpose(img_NIR_HR, [2, 0, 1])

            # pdb.set_trace()

            delta = delta.astype(float) / self.rho
            input1 = torch.from_numpy(input_HR_crop)
            input2 = torch.from_numpy(input_Warp_NIR_HR_crop)

            pts1 = torch.from_numpy(pts1)
            delta = torch.from_numpy(delta)

            img_NIR_HR = torch.from_numpy(img_NIR_HR)
            return input1, input2, pts1, delta, img_NIR_HR,input_HR, input_Warp_NIR_HR


    def __len__(self):
        return len(self.data)
