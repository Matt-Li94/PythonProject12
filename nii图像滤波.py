import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# 读取NII图像
input_image_path = "./nswCHEN_JIAN_CHUN_F_34943_PET_Brain_CIT_OSEM_20221014172616_7.nii"
input_image = sitk.ReadImage(input_image_path)

# 将图像转换为NumPy数组
input_array = sitk.GetArrayFromImage(input_image)

# 自适应邻域加权滤波（ANW）
output_array = np.zeros_like(input_array)

for z in range(input_array.shape[0]):
    # 计算每个切片的局部均值和方差
    local_mean = uniform_filter(input_array[z, :, :], size=3)  # 使用3x3的窗口计算局部均值
    local_variance = uniform_filter((input_array[z, :, :] - local_mean) ** 2, size=3)  # 使用3x3的窗口计算局部方差

    # 计算自适应邻域权重
    k = 200  # 超参数
    alpha = 1.0  # 超参数
    weights = 1.0 / (1.0 + (local_variance / (k * k))) ** alpha

    # 应用自适应邻域权重的滤波
    for i in range(input_array.shape[1]):
        output_array[z, i, :] = np.sum(weights * input_array[z, i, :], axis=0) / np.sum(weights, axis=0)

# 将NumPy数组转换回SimpleITK图像
output_image = sitk.GetImageFromArray(output_array)

# 保存去噪后的图像
output_image_path = 'denoised_image.nii'
sitk.WriteImage(output_image, output_image_path)

# 显示滤波前后的图像
plt.figure(figsize=(12, 6))

# 显示滤波前的图像
plt.subplot(1, 2, 1)
plt.imshow(input_array[50, :, :], cmap='gray')
plt.title('Original Image')

# 显示滤波后的图像
plt.subplot(1, 2, 2)
plt.imshow(output_array[50, :, :], cmap='gray')
plt.title('Denoised Image')

# 保存对比图像
plt.savefig('denoised_comparison.png')

# 显示对比图像
plt.show()
