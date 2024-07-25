#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
遍历整个文件夹，读入二值掩膜图像（灰度图像），
将数值为 255 的像素转换为 RGB 为 [128, 0, 0] 的图像，再保存该图像。

python tools/dataset_converters/binary_mask_to_rgb.py
"""

import os
import cv2
import numpy as np

def convert_mask_to_rgb(input_dir, output_dir):
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        print("Processing: {}".format(filename))

        # 只处理灰度图像文件
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # 读入灰度图像，目标像素的数值为 255
            img_gray = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)
            # print("img_gray.max():", img_gray.max())

            # 将灰度图像转换为 3 通道图像
            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            # 判断是否存在值为 255 的像素
            if np.any(img_gray == [255]):
                # 将像素值为 255 的像素转换为 RGB 的 [128, 0, 0]
                # OpenCV 默认使用 BGR（蓝绿红）颜色空间
                # RGB 颜色 [128, 0, 0] 对应 BGR 颜色 [0, 0, 128]
                img_rgb[np.where(img_gray == [255])] = [0, 0, 128]

            # 保存转换后的图像
            output_filename = os.path.join(output_dir, filename)
            cv2.imwrite(output_filename, img_rgb)

if __name__ == '__main__':
    # 输入文件夹路径和输出文件夹路径
    input_dir = '/Users/grok/Downloads/open-sirst-v2/annotations/masks'
    output_dir = '/Users/grok/Downloads/open-sirst-v2/annotations/rgb_masks'

    # 转换二值掩膜图像为 RGB 图像并保存
    convert_mask_to_rgb(input_dir, output_dir)