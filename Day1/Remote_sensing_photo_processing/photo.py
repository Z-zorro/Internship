import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def read_remote_sensing_image(file_path):
    """
    使用rasterio读取遥感图像文件
    
    参数:
    file_path -- 遥感图像文件路径
    
    返回:
    image_data -- 形状为(H, W, C)的numpy数组
    profile -- 包含地理信息的字典
    """
    try:
        with rasterio.open(file_path) as src:
            bands = []
            for i in range(1, src.count + 1):
                bands.append(src.read(i))
            
            image_data = np.dstack(bands)
            profile = src.profile
            
            print(f"成功读取图像: {file_path}")
            print(f"图像尺寸: {src.height} x {src.width} x {src.count}")
            print(f"数据类型: {src.dtypes[0]}")
            print(f"数值范围: {np.min(image_data)} - {np.max(image_data)}")
            
            return image_data, profile
    except Exception as e:
        print(f"读取失败: {e}")
        raise RuntimeError(f"无法读取遥感图像文件: {file_path}")

def process_remote_sensing(image_data, rgb_indices=(2, 1, 0)):
    """
    处理遥感图像：压缩数据范围并转换为RGB
    
    参数:
    image_data -- 形状为(H, W, C)的numpy数组
    rgb_indices -- 包含RGB波段索引的元组 (红索引, 绿索引, 蓝索引)
    
    返回:
    RGB图像 (NumPy数组，uint8类型，OpenCV BGR顺序)
    """
    if image_data.shape[2] < 3:
        raise ValueError(f"图像至少需要3个波段，当前只有{image_data.shape[2]}个波段")
    
    if max(rgb_indices) >= image_data.shape[2]:
        raise ValueError(f"波段索引超出范围: 最大可用波段索引为{image_data.shape[2]-1}")
    
    clipped = np.clip(image_data, 0, 10000)
    
    compressed = (clipped / 10000.0) * 255.0
    compressed = compressed.astype(np.uint8)
    
    red_band = compressed[:, :, rgb_indices[0]]
    green_band = compressed[:, :, rgb_indices[1]]
    blue_band = compressed[:, :, rgb_indices[2]]
    
    return cv2.merge([blue_band, green_band, red_band])

def save_geotiff_with_rasterio(output_path, rgb_image, profile):
    """
    使用rasterio保存带有地理信息的GeoTIFF文件
    
    参数:
    output_path -- 输出文件路径
    rgb_image -- RGB图像数组 (H, W, 3)，OpenCV BGR顺序
    profile -- 原始图像的地理信息
    """
    profile.update({
        'count': 3,
        'dtype': 'uint8',
        'nodata': None,
        'photometric': 'RGB'
    })
    
    r_band = rgb_image[:, :, 2]
    g_band = rgb_image[:, :, 1]
    b_band = rgb_image[:, :, 0]
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(r_band, 1)
        dst.write(g_band, 2)
        dst.write(b_band, 3)
    
    print(f"已保存GeoTIFF文件: {output_path}")

def visualize_results(rgb_image, original_data, profile, output_dir):
    """
    使用Matplotlib可视化结果
    
    参数:
    rgb_image -- RGB图像 (OpenCV BGR格式)
    original_data -- 原始遥感数据
    profile -- 原始图像的地理信息
    output_dir -- 指定的输出文件夹路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title('压缩后的RGB图像')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    rgb_original = original_data[:, :, :3].astype(float) / 10000.0
    plt.imshow(rgb_original)
    plt.title('原始RGB波段（归一化）')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    if original_data.shape[2] >= 4:
        nir = original_data[:, :, 3].astype(float) / 10000.0
        plt.imshow(nir, cmap='viridis')
        plt.title('近红外波段')
        plt.colorbar()
    else:
        plt.text(0.5, 0.5, '无近红外数据', ha='center', va='center')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    if original_data.shape[2] >= 5:
        swir = original_data[:, :, 4].astype(float) / 10000.0
        plt.imshow(swir, cmap='hot')
        plt.title('短红外波段')
        plt.colorbar()
    else:
        plt.text(0.5, 0.5, '无短红外数据', ha='center', va='center')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    colors = ['blue', 'green', 'red']
    for i, color in enumerate(colors):
        band = rgb_image[:, :, i].ravel()
        plt.hist(band, bins=256, range=(0, 256), color=color, alpha=0.5, label=f'{color}波段')
    plt.title('RGB直方图')
    plt.xlabel('像素值')
    plt.ylabel('频率')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    band_names = ['蓝', '绿', '红', '近红外', '短红外']
    for i in range(min(5, original_data.shape[2])):
        band = original_data[:, :, i].ravel()
        band = band[band <= 10000]
        plt.hist(band, bins=50, range=(0, 10000), alpha=0.5, label=f'{band_names[i]}波段')
    plt.title('原始数据分布')
    plt.xlabel('像素值')
    plt.ylabel('频率')
    plt.legend()
    
    plt.tight_layout()
    
    output_path_analysis = os.path.join(output_dir, "remote_sensing_analysis.jpg")
    plt.savefig(output_path_analysis, dpi=150, bbox_inches='tight')
    print(f"已保存分析图像: {output_path_analysis}")
    plt.close()
    
    # 使用Matplotlib显示带地理信息的图像
    rgb_for_display = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_for_display)
    plt.title('压缩后的RGB图像（带地理信息）')
    plt.axis('off')
    
    output_path_geo = os.path.join(output_dir, "rgb_geo.png")
    plt.savefig(output_path_geo, dpi=150, bbox_inches='tight')
    print(f"已保存地理信息图像: {output_path_geo}")
    plt.close()

def main():
    input_file = "Day1\\Remote_sensing_photo_processing\\photo.tif"
    output_dir = "Day1\\Remote_sensing_photo_processing"
    rgb_output = os.path.join(output_dir, "output_rgb.tif")
    jpg_output = os.path.join(output_dir, "output_rgb.jpg")
    rgb_indices = (2, 1, 0)
    
    try:
        image_data, profile = read_remote_sensing_image(input_file)
        rgb_image = process_remote_sensing(image_data, rgb_indices)
        save_geotiff_with_rasterio(rgb_output, rgb_image, profile)
        cv2.imwrite(jpg_output, rgb_image)
        print(f"已保存JPG文件: {jpg_output}")
        
        visualize_results(rgb_image, image_data, profile, output_dir)
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()