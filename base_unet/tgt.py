import numpy as np
import cv2
import os

input_dir = "../NEU_Seg-main/NEU_Seg-main/annotations/test"
output_dir = 'test_ground_truth'
os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # 构建完整的输入文件路径
        input_path = os.path.join(input_dir, filename)

        # 读取 PNG 图像
        image = cv2.imread(input_path)
        image_resized = cv2.resize(image, (224, 224))
        image_array = np.array(image_resized)

        # 构建输出文件路径
        output_filename = filename.replace('.png', '.npy')
        output_path = os.path.join(output_dir, output_filename)

        # 保存为 .npy 文件

        np.save(output_path, image_array)

        print(f"Saved {input_path} as {output_path}")

print("All PNG files have been converted to .npy files.")