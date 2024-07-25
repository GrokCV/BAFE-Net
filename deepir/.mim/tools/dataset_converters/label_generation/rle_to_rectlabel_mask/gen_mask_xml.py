#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python tools/dataset_converters/label_generation/rle_to_mask/gen_mask_xml.py
"""

import os
from PIL import Image
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def create_xml(img_path, img_filename, width, height):
    annotation = Element("annotation")

    folder = SubElement(annotation, "folder")
    folder.text = "PNGImages"

    filename = SubElement(annotation, "filename")
    filename.text = img_filename

    size = SubElement(annotation, "size")
    width_elem = SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = SubElement(size, "height")
    height_elem.text = str(height)
    depth = SubElement(size, "depth")
    depth.text = "3"

    obj = SubElement(annotation, "object")
    name = SubElement(obj, "name")
    name.text = "Sky"
    difficult = SubElement(obj, "difficult")
    difficult.text = "0"

    pixels = SubElement(obj, "pixels")
    id = SubElement(pixels, "id")
    id.text = "0"

    bndbox = SubElement(obj, "bndbox")
    xmin = SubElement(bndbox, "xmin")
    xmin.text = "1"
    ymin = SubElement(bndbox, "ymin")
    ymin.text = "1"
    xmax = SubElement(bndbox, "xmax")
    xmax.text = str(width - 1)
    ymax = SubElement(bndbox, "ymax")
    ymax.text = str(height - 1)

    xml_str = tostring(annotation)
    xml_pretty_str = parseString(xml_str).toprettyxml(indent="    ")

    xml_filename = os.path.splitext(img_filename)[0] + ".xml"
    xml_path = os.path.join(img_path, xml_filename)

    with open(xml_path, "w") as f:
        f.write(xml_pretty_str)


if __name__ == "__main__":
    img_folder = "/Users/grok/Nutstore Files/codes/SIRSTdevkit/SkySeg/Mask"

    for img_filename in os.listdir(img_folder):
        if img_filename.endswith(".png"):
            img_path = os.path.join(img_folder, img_filename)
            img = Image.open(img_path)
            width, height = img.size

            create_xml(img_folder, img_filename, width, height)
        # break