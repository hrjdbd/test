import numpy as np
import rasterio

# image1 = r"E:\A_workspace\8.DRA\03\01_LC08_L2SP_040035_20140829_20200911_02_T1_ST_B10_clip_BM_100M_upscaling.tif"
# image1 = r"E:\A_workspace\9.VER\test\07-1KM\01_LC08_L2SP_192030_20170408_20200904_02_T1_ST_B10_clip_BM_100M_upscaling.tif"
image1 = r"E:\A_workspace\9.VER\MODIS_1KM_BM\02_MOD11A1.A2013231.LST_Day_1km_clip_BM.tif"
image2 = r"E:\A_workspace\9.VER\07-1KM\01_MOD11A1.A2017098.LST_Day_1km_clip_BM.tif"
# image2 = r"E:\A_workspace\9.VER\07-1KM\新建文件夹\01_LC08_L2SP_192030_20170408_20200904_02_T1_ST_B10_2013231_FSDAF.tif"



def cov(im1,im2):
        mean1 = np.mean(im1)
        mean2 = np.mean(im2)
        num1 = im1.shape[0]
        num1 = im1.shape[1]
        sum = 0
        for i in range(im1.shape[0]):
                for j in range(im1.shape[1]):
                        sum = sum+(im1[i,j]-mean1)*(im2[i,j]-mean2)
        cov = sum/((im1.shape[0]*im1.shape[1])-1)
        # %协方差
        return cov

def ssim(src, dst):
    
    # 数据准备
    # 均值mean
    mean_src = np.mean(src)
    mean_dst = np.mean(dst)
    # 方差var
    var_src = np.var(src)
    var_dst = np.var(dst)
    cov1 = cov(src, dst)
    # 标准差std
    std_src = np.std(src)
    std_dst = np.std(dst)
    cc = cov1/(std_dst * std_src)
    # 常数c1,c2,c3
    K1 = 0.01
    K2 = 0.03
    L = 255
    c1 = (K1*L)**2
    c2 = (K2*L)**2
    c3 = c2 / 2
 
        
    # 计算ssim
    l = (2*mean_src*mean_dst + c1)/(mean_src**2 + mean_dst**2 + c1)
    c = (2*var_src*var_dst + c2)/(var_src**2 + var_dst**2 + c2)
    s = (cov1 + c3)/(var_src*var_dst + c3)
    
    ssim = l * c * s
    mae = np.sum(np.abs(src-dst))/src.size
    return ssim,cc,mae

def RMSE(im1,im2):
        mse = np.sum(pow((im1 - im2),2)) / np.size(im1)
        rmse = np.sqrt(mse)
        return rmse

with rasterio.open(str(image1)) as ds:
        im1 = ds.read().astype(np.float32)
        im1 = np.squeeze(im1,0)
        # max1 = im1.max()                      
        # min1 = im1.min()
        # im1 = (im1 - min1)/(max1 - min1)
with rasterio.open(str(image2)) as ds:
        im2 = ds.read().astype(np.float32)
        im2 = np.squeeze(im2,0)
        # max2 = im2.max()
        # min2 = im2.min()
        # im2 = (im2- min2)/(max2 - min2)
# print("SSIM:" + str(ssim(im1,im2)[0]))
print("CC:" + str(ssim(im1,im2)[1]))
print("RMSE:" + str(RMSE(im1,im2)))
print("MAE:" + str(ssim(im1,im2)[2]))