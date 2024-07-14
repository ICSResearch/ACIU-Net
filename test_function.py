import os
import cv2
import copy
from time import time
import math
from skimage.metrics import structural_similarity as ssim
import torch
import numpy as np




class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, full_mask, flag=True):
        full_mask = full_mask[..., 0]
        x_in_k_space = torch.fft.fft2(x)
        masked_x_in_k_space = x_in_k_space * full_mask.view(1, 1, *(full_mask.shape))

        masked_x = torch.abs(torch.fft.ifft2(masked_x_in_k_space))


        return masked_x
##########################################################################################
def compute_measure(y_gt, y_pred, data_range):
    pred_rmse = compute_RMSE(y_pred, y_gt)
    return (pred_rmse)

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))
#####################################################################################
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#####################################################################################
# test implement
def test_implement_MRI(filepaths, model, cs_ratio, mask, test_name, epoch_num, result_dir,run_mode,device):
    print('\n')
    print("CS Reconstruction Start")
    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    RMSE_total = []
    model.eval()
    with torch.no_grad():
        for img_no in range(ImgNum):
            imgName = filepaths[img_no]
            Iorg = cv2.imread(imgName, 0)
            Icol = Iorg.reshape(1, 1, 256, 256) / 255.0
            Img_output = Icol
            start = time()
            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            PhiTb = FFT_Mask_ForBack()(batch_x, mask)
            x_output = model(PhiTb, mask)
            end = time()
            Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)
            X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)
            rec_PSNR = psnr(X_rec * 255., Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec * 255., Iorg.astype(np.float64), data_range=255)
            m_reg = compute_measure(Iorg.astype(np.float64)/255, X_rec, 1)
            RMSE_total.append(m_reg)
            if run_mode == 'test':
                print("[%02d/%02d] Run time for %s is %.4f, Proposed PSNR is %.2f, Proposed SSIM is %.4f, Proposed RMSE is %.4f" % (
                img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM,m_reg))
                im_rec_rgb = np.clip(X_rec * 255, 0, 255).astype(np.uint8)
                img_name = imgName.split('/')
                img_name = str(img_name[2])
                img_name = img_name.split('.')
                img_dir = result_dir + img_name[0] + "_PSNR_%.2f_SSIM_%.4f_RMSE_%.4f.png" % (rec_PSNR, rec_SSIM,m_reg)
                cv2.imwrite(img_dir, im_rec_rgb)
            del x_output
            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
    print('\n')
    output_data = "CS ratio is %d, Avg Proposed PSNR/SSIM/RMSE for %s is %.2f-%.2f/%.4f-%.4f/%.4f-%.4f, Epoch number of model is %d \n" % (
        cs_ratio, test_name, np.mean(PSNR_All), np.std(PSNR_All), np.mean(SSIM_All), np.std(SSIM_All),
        np.array(RMSE_total).mean(), np.array(RMSE_total).std(), epoch_num)
    print(output_data)
    print("CS Reconstruction End")
    return np.mean(PSNR_All), np.mean(SSIM_All),np.array(RMSE_total).mean()
##############################################################################################################