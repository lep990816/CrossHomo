import numpy as np
import os
import cv2

def save_correspondences_img_ch3(img1, img2, corr1, corr2, pred_corr2, results_dir, img_name):
    """ Save pair of images with their correspondences into a single image. Used for report"""
    new_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1],3), np.uint8)
    new_img[0:img1.shape[0], 0:img1.shape[1],:] = img1.copy()
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1],:] = img2.copy()
    # pdb.set_trace()
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
    # cv2.polylines(img2, np.int32([corr2]), 1, (188, 50, 0), 5, cv2.LINE_AA)
    cv2.polylines(img2, np.int32([corr2]), 1, (30, 60, 250), 7, cv2.LINE_AA)

    # cv2.polylines(img2, np.int32([pred_corr2]), 1, (255,60,30), 4, cv2.LINE_AA)
    cv2.polylines(img2, np.int32([pred_corr2]), 1, (10,248,250), 4, cv2.LINE_AA)

    # corr2_ = (corr2 + np.array([img1.shape[1], 0])).astype(np.int32)
    # pred_corr2_ = (pred_corr2 + np.array([img1.shape[1], 0])).astype(np.int32)

    # cv2.polylines(new_img, np.int32([corr2_]), 1, (0, 225, 0), 3, cv2.LINE_AA)
    # # cv2.polylines(new_img, np.int32([pred_corr2_]), 1, (0, 225, 0), 2, cv2.LINE_AA)
    # cv2.polylines(new_img, np.int32([pred_corr2_]), 1, (0, 225, 225), 3, cv2.LINE_AA)


    # Save image
    visual_file_name = os.path.join(results_dir, img_name)
    # cv2.putText(full_stack_images, 'RMSE %.2f'%h_loss,(800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
    cv2.imwrite(visual_file_name, img2)
    print('Wrote file %s' % visual_file_name)

def save_correspondences_img_ch3_ori(img1, img2, corr1, corr2, pred_corr2, results_dir, img_name):
    """ Save pair of images with their correspondences into a single image. Used for report"""
    new_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1],3), np.uint8)
    new_img[0:img1.shape[0], 0:img1.shape[1],:] = img1.copy()
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1],:] = img2.copy()
    # pdb.set_trace()
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)

    cv2.polylines(new_img, np.int32([corr1]), 1, (0, 225, 0), 3, cv2.LINE_AA)

    corr2_ = (corr2 + np.array([img1.shape[1], 0])).astype(np.int32)
    pred_corr2_ = (pred_corr2 + np.array([img1.shape[1], 0])).astype(np.int32)

    cv2.polylines(new_img, np.int32([corr2_]), 1, (0, 225, 0), 3, cv2.LINE_AA)
    # cv2.polylines(new_img, np.int32([pred_corr2_]), 1, (0, 225, 0), 2, cv2.LINE_AA)
    cv2.polylines(new_img, np.int32([pred_corr2_]), 1, (0, 225, 225), 3, cv2.LINE_AA)


    # Save image
    visual_file_name = os.path.join(results_dir, img_name)
    # cv2.putText(full_stack_images, 'RMSE %.2f'%h_loss,(800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
    cv2.imwrite(visual_file_name, new_img)
    print('Wrote file %s' % visual_file_name)