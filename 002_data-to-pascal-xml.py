#!/usr/bin/env python3


"""
This script crawls over 9263 training images and 1873 items
On my Macbook pro this takes: 4 minutes

"""
import cv2
import os
import numpy as np
from glob import iglob # python 3.5 or newer
from shutil import copyfile


# The script
curr_path = os.getcwd()

import xml.etree.cElementTree as ET

# settings
cnt = 0
hog = cv2.HOGDescriptor((80, 80), (16, 16), (8,8), (8,8), 9)
# data = []
# label = []


def newXMLPASCALfile(imageheight, imagewidth, path, basename):
    # print(filename)
    annotation = ET.Element("annotation", verified="yes")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = basename
    ET.SubElement(annotation, "path").text = path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "test"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(imagewidth)
    ET.SubElement(size, "height").text = str(imageheight)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    tree = ET.ElementTree(annotation)
    # tree.write("filename.xml")
    return tree

def appendXMLPASCAL(curr_et_object,x1, y1, w, h, filename):
    et_object = ET.SubElement(curr_et_object.getroot(), "object")
    ET.SubElement(et_object, "name").text = "face"
    ET.SubElement(et_object, "pose").text = "Unspecified"
    ET.SubElement(et_object, "truncated").text = "0"
    ET.SubElement(et_object, "difficult").text = "0"
    bndbox = ET.SubElement(et_object, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(x1)
    ET.SubElement(bndbox, "ymin").text = str(y1)
    ET.SubElement(bndbox, "xmax").text = str(x1+w)
    ET.SubElement(bndbox, "ymax").text = str(y1+h)
    filename = filename.strip().replace(".jpg",".xml")
    curr_et_object.write(filename)
    return curr_et_object




def readAndWrite(bbx_gttxtPath):
    cnt = 0
    with open(bbx_gttxtPath, 'r') as f:
        curr_img = ''

        curr_filename = ""
        curr_path = ""

        curr_et_object = ET.ElementTree()


        img = np.zeros((80, 80))
        for line in f:
            inp = line.split(' ')

            # if line.find("--") != -1:
            #     curr_filename = line.split('--')[1]
            #     # reset elements
            #     # emptyEl = ET.Element("")
            #     curr_et_object = ET.ElementTree()

            if len(inp)==1:
                img_path = inp[0]
                img_path = img_path[:-1]
                curr_img = img_path
                if curr_img.isdigit():
                    continue
                # print(Train_path+'/'+curr_img)
                img = cv2.imread(Train_path + '/' + curr_img, 2) # POSIX only
                # print( len(list(curr_et_object.getroot()) )  )
                curr_filename = curr_img.split("/")[1].strip()
                curr_path = os.path.join(Train_path, os.path.dirname(curr_img))
                curr_et_object = newXMLPASCALfile(img.shape[0],img.shape[1],curr_path, curr_filename )
                # print( curr_et_object  )

            else:
                # print(img)
                inp = [int(i) for i in inp[:-1]]
                x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = inp
                n = max(w,h)
                if invalid == 1 or blur > 0 or n < 50:
                    continue
                img2 = img[y1:y1+n, x1:x1+n]
                img3 = cv2.resize(img2, (80, 80))
                vec = hog.compute(img3)
                # data.append(vec)
                # label.append(1)
                cnt += 1

                fileNow = os.path.join(curr_path,curr_filename)
                print("{}: {} {} {} {}".format(len(vec),x1, y1, w, h) + " " + fileNow)

                curr_et_object = appendXMLPASCAL(curr_et_object,x1, y1, w, h, fileNow )


# ################################ TRAINING DATA 9263 ITEMS ##################################
# # # Run Script for Training data
Train_path = os.path.join(curr_path, "data", "WIDER_train", "images" )
## comment this out
bbx_gttxtPath = os.path.join(curr_path, "data", "wider_face_split", "wider_face_train_bbx_gt.txt" )
readAndWrite(bbx_gttxtPath)


# To folders:
to_xml_folder = os.path.join(curr_path, "data", "tf_wider_train", "annotations", "xmls" )
to_image_folder = os.path.join(curr_path, "data", "tf_wider_train", "images" )

# make dir => wider_data in folder
try:
    os.makedirs(to_xml_folder)
    os.makedirs(to_image_folder)
except Exception as e:
    pass

rootdir_glob = Train_path + '/**/*' # Note the added asterisks # This will return absolute paths
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f)]

train_annotations_index = os.path.join(curr_path, "data", "tf_wider_train", "annotations", "train.txt" )

with open(train_annotations_index, "a") as indexFile:
    for f in file_list:
        if ".xml" in f:
            print(f)
            copyfile(f, os.path.join(to_xml_folder, os.path.basename(f) ))
            img = f.replace(".xml",".jpg")
            copyfile(img, os.path.join(to_image_folder, os.path.basename(img) ))
            indexFile.write(os.path.basename(f.replace(".xml","")) + "\n")


################################ VALIDATION DATA 1873 ITEMS ##################################

# Run Script for Validation data
Train_path = os.path.join(curr_path, "data", "WIDER_val", "images" )
bbx_gttxtPath = os.path.join(curr_path, "data", "wider_face_split", "wider_face_val_bbx_gt.txt" )
readAndWrite(bbx_gttxtPath)


# To folders:
to_xml_folder = os.path.join(curr_path, "data", "tf_wider_val", "annotations", "xmls" )
to_image_folder = os.path.join(curr_path, "data", "tf_wider_val", "images" )

# make dir => wider_data in folder
try:
    os.makedirs(to_xml_folder)
    os.makedirs(to_image_folder)
except Exception as e:
    pass


rootdir_glob = Train_path + '/**/*' # Note the added asterisks # This will return absolute paths
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f)]

train_annotations_index = os.path.join(curr_path, "data", "tf_wider_val", "annotations", "val.txt" )

with open(train_annotations_index, "a") as indexFile:
    for f in file_list:
        if ".xml" in f:
            print(f)
            copyfile(f, os.path.join(to_xml_folder, os.path.basename(f) ))
            img = f.replace(".xml",".jpg")
            copyfile(img, os.path.join(to_image_folder, os.path.basename(img) ))
            indexFile.write(os.path.basename(f.replace(".xml","")) + "\n")
