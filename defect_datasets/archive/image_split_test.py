import cv2
from xml.dom import minidom
import os
raw_img = cv2.imread("./images/img_02_SIS001491_00280.jpg")
xml_name = "img_02_SIS001491_00280.xml"
xmldoc = minidom.parse("./label/label/img_02_SIS001491_00280.xml")
raw_h, raw_w, _ = raw_img.shape
width_split = 4
height_split = 2

expect_size = 640
width_step = 512
height_step = 500


## image split coordinates
image_split_coord_list = []
for width_index in range(width_split):
    for height_index in range(height_split):
        width_start =width_index * width_step
        if (width_start + expect_size) > raw_w:
            width_end = raw_w
            width_start = raw_w - expect_size
        else:
            width_end = width_start + expect_size

        height_start = height_index * height_step

        if (height_start + expect_size) > raw_h:
            height_end = raw_h
            height_start = raw_h - expect_size
        else:
            height_end = height_start + expect_size

        image_split_coord_list.append((width_start, height_start, width_end, height_end))

## object box ordinate processing and converting
itemlist = xmldoc.getElementsByTagName('object')

for img_index, (width_start, height_start, width_end, height_end) in enumerate(image_split_coord_list):

    yolo_obj_list = []

    for item in itemlist:
        xmin = int(((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data)
        ymin = int(((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data)
        xmax = int(((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data)
        ymax = int(((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data)

        if width_end < xmin:
            continue
        if height_end < ymin:
            continue
        if xmax < width_start:
            continue
        if ymax < height_start:
            continue

        new_xmin = max(width_start, xmin) - width_start
        new_ymin = max(height_start, ymin) - height_start
        new_xmax = min(width_end, xmax) - width_start
        new_ymax = min(height_end, ymax) - height_start

        obj_x_min, obj_w, obj_y_min, obj_h = new_xmin, (new_xmax - new_xmin), new_ymin, (new_ymax - new_ymin)

        yolo_center_x = round(float((obj_x_min + obj_w / 2.0) / expect_size), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / expect_size), 6)
        yolo_w = round(float(obj_w / expect_size), 6)
        yolo_h = round(float(obj_h / expect_size), 6)

        classid = (item.getElementsByTagName('name')[0]).firstChild.data
        tmp_label = classid.split('_')

        # label_id = self._label_id_map[tmp_label[1]]
        label_id =  int(tmp_label[0])

        yolo_obj_list.append((label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h))

    split_img = raw_img[height_start:height_end, width_start:width_end]

    img_path = "./image_split_test_dir/" + xml_name.replace('.xml', '') + "_split_" + str(img_index) + ".jpg"

    cv2.imwrite(img_path, split_img)

    txt_path = os.path.join(img_path.replace('.jpg', '.txt'))

    with open(txt_path, 'w+') as f:
        for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
            yolo_obj_line = '%s %s %s %s %s\n' % yolo_obj \
                if yolo_obj_idx + 1 != len(yolo_obj_list) else \
                '%s %s %s %s %s' % yolo_obj
            f.write(yolo_obj_line)





