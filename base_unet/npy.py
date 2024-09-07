import os
import cv2
import numpy as np



input_dir = 'inference_results'
output_dir = 'inference_results_npy'
# output_dir = r'D:\AILAB\校园算法\挑战赛\赛题2-钢材表面缺陷检测与分割\Seg（分割赛题记分）\test_predictions'
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