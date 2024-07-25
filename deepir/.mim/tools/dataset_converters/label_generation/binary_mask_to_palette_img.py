#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
遍历整个文件夹，读入二值掩膜图像，将灰度为 255 的像素转换为调色盘图像并保存。

- 调色盘采用 PASCAL VOC 的调色盘，目标类即为 `[128, 0, 0]`

python tools/dataset_converters/label_generation/binary_mask_to_palette_img.py
"""

import os
import cv2
import numpy as np
from PIL import Image

def get_palette(palette_path):
    # 读取调色板图像文件
    palette = Image.open(palette_path)

    # 获取调色板
    palette_colors = palette.getpalette()
    return palette_colors

def convert_mask_to_palette(input_dir, output_dir, palette_colors):
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):

        # 只处理灰度图像文件
        if filename.endswith('.png') or filename.endswith('.jpg'):
            print("Processing: {}".format(filename))

            # 读入灰度图像，目标像素的数值为 255
            img_gray = cv2.imread(os.path.join(input_dir, filename),
                                  cv2.IMREAD_GRAYSCALE)
            img_palette = img_gray

            # 转换成调色板图像
            # 判断是否存在值为 255 的像素
            if np.any(img_gray == [255]):
                # 将像素值为 255 的像素转换为索引值  1
                img_palette[np.where(img_gray == [255])] = 1
            img_palette = Image.fromarray(img_palette.astype(np.uint8), mode="P")
            img_palette.putpalette(palette_colors)

            # 保存调色板图像
            output_filename = os.path.join(output_dir, filename)
            img_palette.save(output_filename)

def check_image_type(output_dir):
    for filename in os.listdir(output_dir):
        # 只处理灰度图像文件
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(output_dir, filename)
            with Image.open(image_path) as img:
                # 获取图像模式，如果是“P”表示是调色板图像，如果是“RGB”则是 RGB 图像
                img_mode = img.mode

                if img_mode == "P":
                    print(f"{image_path} 是调色板图像")
                elif img_mode == "RGB":
                    print(f"{image_path} 是 RGB 图像")
                else:
                    print(f"{image_path} 不是支持的图像类型")

if __name__ == '__main__':
    # 调色盘图像路径
    palette_path = 'tools/dataset_converters/label_generation/2007_000346.png'

    # 输入文件夹路径和输出文件夹路径
    # input_dir = '/Users/grok/Downloads/open-sirst-v2/annotations/masks'
    # output_dir = '/Users/grok/Downloads/open-sirst-v2/annotations/palette_masks'
    input_dir = '/Users/grok/Nutstore Files/codes/SIRSTdevkit/SkySeg/BinaryMask'
    output_dir = '/Users/grok/Nutstore Files/codes/SIRSTdevkit/SkySeg/PaletteMask'

    # 读取 PASCAL VOC 的调色盘图像，获取调色盘
    palette_colors = get_palette(palette_path)

    # 转换二值掩膜图像为 RGB 图像并保存
    convert_mask_to_palette(input_dir, output_dir, palette_colors)

    # 检查图像类型
    check_image_type(output_dir)