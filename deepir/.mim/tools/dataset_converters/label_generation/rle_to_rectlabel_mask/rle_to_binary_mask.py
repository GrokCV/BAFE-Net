#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python tools/dataset_converters/label_generation/rle_to_mask/rle_to_binary_mask.py
"""

import json
import numpy as np
from PIL import Image
from label_studio_converter import brush

if __name__ == "__main__":
    json_paths = [
        "/Users/grok/Downloads/SkyJSON/1/project-3-at-2023-06-08-22-07-0578db3d.json",
        "/Users/grok/Downloads/SkyJSON/2/project-5-at-2023-06-14-00-45-9f9eda60.json",
        "/Users/grok/Downloads/SkyJSON/3/project-6-at-2023-06-08-22-13-9f868850.json",
        "/Users/grok/Downloads/SkyJSON/4/project-19-at-2023-06-14-12-23-cc6fc071.json",
        "/Users/grok/Downloads/SkyJSON/5/project-6-at-2023-06-08-23-03-297b6558.json",
        "/Users/grok/Downloads/SkyJSON/5/project-9-at-2023-06-08-23-04-037c0f27.json",
        "/Users/grok/Downloads/SkyJSON/6/project-2-at-2023-06-09-12-58-f2f794a8.json",
        "/Users/grok/Downloads/SkyJSON/7/project-1-at-2023-06-08-22-22-a5c173c8.json",
        ]
    base_path = "/Users/grok/Downloads/SkyJSON/"
    mask_dir = "/Users/grok/Downloads/SkyMask"
    merge_masks = True
    mask_num = 0
    for json_path in json_paths:
        assignee_id = json_path[len(base_path):].split('/', 1)[0]
        with open(json_path, "r") as file:
            json_data = json.load(file)
            print(len(json_data))
            mask_num += len(json_data)
            for item in json_data:
                filename = item['file_upload'].split("-", 1)[-1]
                result_len = len(item['annotations'][0]['result'])
                if result_len == 0:
                    print(f"no mask: {assignee_id} - {filename}")
                    continue

                # convert RLE to mask image
                masks = []
                for idx in range(result_len):
                    result = item['annotations'][0]['result'][idx]
                    out = brush.decode_rle(
                        result['value']['rle'], print_params=False)
                    height = result['original_height']
                    width = result['original_width']
                    mask = np.reshape(out, [height, width, 4])[:, :, 3]
                    masks.append(mask > 128)

                # save mask image
                mask_path = mask_dir + "/" + filename
                if merge_masks:
                    # merge masks
                    merged_mask = np.zeros_like(masks[0], dtype=bool)
                    for mask in masks:
                        merged_mask = np.logical_or(merged_mask, mask)
                    mask_image = Image.fromarray(
                        (merged_mask * 255).astype(np.uint8), mode="L")
                    mask_image.save(mask_path)
                else:
                    for idx, mask in enumerate(masks):
                        mask_image = Image.fromarray(
                            ((mask > 128) * 255).astype(np.uint8), mode="L")
                        if idx > 0:
                            print(f"multi mask: {assignee_id} - {filename}")
                            mask_path = mask_path.replace(".png", f"_{idx+1}.png")
                        mask_image.save(mask_path)
    print("total mask num: ", mask_num)