import os
import cv2
import random
import numpy as np
from numpy.linalg import inv
import time
import pdb

train_path = '/temp_disk2/lep/dataset/DPDN/train/'
# train_path = '/raid/yangwenzhe/lep_code/dataset/DPDN/train'
val_path = '/temp_disk2/lep/dataset/DPDN/test/'
test_path = '/temp_disk2/lep/dataset/DPDN/test/'
modal1_path = '/temp_disk2/lep/SR_Homo/Data/ori/'
modal2_path = '/temp_disk2/lep/SR_Homo/Data/ori/'


def ImagePreProcessing(source_path,image_number, rho, patch_size, imsize, types):
    # if image_number[0] == '0':
    #     image_number_modal1 = image_number[1:]
    # else:
    # image_number_modal1 = image_number
    image_name1 = 'Depth'

    image_path_1 = os.path.join(source_path, 'modal2', 'Depth_' + image_number + '.bmp')
    # image_path_1 = os.path.join(source_path, 'modal1', 'RGB_' + image_number + '.png')

    image_path_2 = os.path.join(source_path, 'modal1', 'RGB_' + image_number + '.png')
    # pdb.set_trace()
    img1 = cv2.imread(image_path_1)
    img1 = cv2.resize(img1, imsize)

    img2 = cv2.imread(image_path_2)
    img2 = cv2.resize(img2, imsize)

    
    position_p = (random.randint(rho, imsize[0] - rho - patch_size), random.randint(rho, imsize[1] - rho - patch_size))
    position_p = (64,64)
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

    # Extract image patches (not stored)
    # Ip1 = test_image[tl_point[1]:br_point[1], tl_point[0]:br_point[0]]
    # Ip2 = warped_image[tl_point[1]:br_point[1], tl_point[0]:br_point[0]]

    training_image = np.dstack((img2, warped_image))
    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    datum = (training_image, np.array(four_points), H_four_points, img1)

    return datum


# save .npy files
def savedata(source_path, new_path, rho, patch_size, imsize, data_size, types):
    modal1_path = os.path.join(source_path, 'modal2')
    modal2_path = os.path.join(source_path, 'modal1')
    lst_1 = os.listdir(modal1_path + '/')
    lst_2 = os.listdir(modal2_path + '/')
    filenames1 = [os.path.join(modal1_path, l) for l in lst_1 if l[-3:] == 'bmp']
    filenames2 = [os.path.join(modal2_path, l) for l in lst_2 if l[-3:] == 'png']
    # pdb.set_trace()
    print("Generate {} {} files from {} raw data...".format(data_size, new_path, len(filenames2)))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for i in range(data_size):
        image_path1 = random.choice(filenames2)
        # image_path1 = filenames2[i]
        image_name = image_path1.split("/")[-1]

        image_number = image_name.split(".")[0]
        image_number = image_number.split("_")[-1]
        # pdb.set_trace()
        np.save(new_path + '/' + ('%s' % i).zfill(6), ImagePreProcessing(source_path,image_number, rho, patch_size, imsize, types))
        if (i + 1) % 1000 == 0:
            print('--image number ', i + 1)


if __name__ == "__main__":
    start = time.time()
    # rho = 56
    # patch_size = 448
    # imsize = (706, 706)
    # rho = 24
    # patch_size = 224
    # imsize = (320, 320)
    rho = 48
    patch_size = 384
    imsize = (512, 512)
    savedata(train_path, '/temp_disk2/lep/SR_Homo/Data/DPDN_RGBonly/training/', rho, patch_size, imsize, data_size=10000, types='train')
    savedata(val_path, '/temp_disk2/lep/SR_Homo/Data/DPDN_RGBonly/validation/', rho, patch_size, imsize, data_size=100, types='test')
    savedata(val_path, '/temp_disk2/lep/SR_Homo/Data/DPDN_RGBonly/testing/', rho, patch_size, imsize, data_size=100, types='test')
    # savedata(val_path, '/temp_disk2/lep/SR_Homo/Data/DPDN_384/testing_5/', rho, patch_size, imsize, data_size=50, types='train')

    elapsed_time = time.time() - start
    print("Generate dataset in {:.0f}h {:.0f}m {:.0f}s.".format(
        elapsed_time // 3600, (elapsed_time % 3600) // 60, (elapsed_time % 3600) % 60))

    # # show sample
    # from matplotlib import pyplot as plt
    # npy = random.choice([os.path.join('./training/', f) for f in os.listdir('./training/')])
    # ori_images, pts1, delta = np.load(npy, allow_pickle=True)
    # pts2 = pts1 + delta
    # patch1 = ori_images[:, :, 0].copy()
    # patch2 = ori_images[:, :, 1].copy()
    # patch1 = cv2.cvtColor(patch1, cv2.COLOR_GRAY2RGB)
    # patch2 = cv2.cvtColor(patch2, cv2.COLOR_GRAY2RGB)
    # cv2.polylines(patch1, [pts1], True, (81, 167, 249), 2, cv2.LINE_AA)
    # cv2.polylines(patch1, [pts2], True, (111, 191, 64), 2, cv2.LINE_AA)
    # cv2.polylines(patch2, [pts1], True, (111, 191, 64), 2, cv2.LINE_AA)
    # plt.subplot(121), plt.imshow(patch1)
    # plt.subplot(122), plt.imshow(patch2)
    # plt.show()