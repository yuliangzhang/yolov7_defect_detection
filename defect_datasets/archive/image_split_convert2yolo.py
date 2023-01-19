'''
Created on Aug 18, 2021
@author: xiaosonh
'''
import os
import sys
import argparse
import shutil
import math
from collections import OrderedDict
from shutil import copyfile
# import xml
import cv2
import PIL.Image
from xml.dom import minidom

from sklearn.model_selection import train_test_split
from labelme import utils


class Xml2YOLO(object):

    def __init__(self, xml_dir, targe_height=640, target_width=640):
        self._xml_dir = xml_dir
        self._xml_label_dir = os.path.join(self._xml_dir, 'label/label')
        self._xml_image_dir = os.path.join(self._xml_dir, 'images')
        self._target_height = targe_height
        self._target_width = target_width

        self._label_id_map = self._get_label_id_map(self._xml_label_dir)

    def _make_train_val_dir(self):
        self._label_dir_path = os.path.join(self._xml_dir,
                                            'YOLODataset/labels/')
        self._image_dir_path = os.path.join(self._xml_dir,
                                            'YOLODataset/images/')

        for yolo_path in (os.path.join(self._label_dir_path + 'train/'),
                          os.path.join(self._label_dir_path + 'val/'),
                          os.path.join(self._image_dir_path + 'train/'),
                          os.path.join(self._image_dir_path + 'val/')):
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)

            os.makedirs(yolo_path)

    def _get_label_id_map(self, xml_dir):
        label_dic = dict()

        for file_name in os.listdir(xml_dir):
            if file_name.endswith('xml'):
                xml_path = os.path.join(xml_dir, file_name)
                # print("file_name:", file_name)
                try:
                    xmldoc = minidom.parse(xml_path)
                    itemlist = xmldoc.getElementsByTagName('object')
                    for item in itemlist:
                        classid = (item.getElementsByTagName('name')[0]).firstChild.data
                        tmp_label = classid.split('_')
                        if 'yaozhe' in tmp_label[1]:
                            label_dic['yaozhe'] = int(tmp_label[0]) - 1
                        else:
                            label_dic[tmp_label[1]] = int(tmp_label[0]) - 1
                except Exception:
                    print("Open Error Filename: ", xml_path)

        return label_dic

    def _train_test_split(self, folders, xml_names, val_size):
        if len(folders) > 0 and 'train' in folders and 'val' in folders:
            train_folder = os.path.join(self._xml_dir, 'train/')
            train_xml_names = [train_sample_name + '.xml' \
                                for train_sample_name in os.listdir(train_folder) \
                                if os.path.isdir(os.path.join(train_folder, train_sample_name))]

            val_folder = os.path.join(self._xml_dir, 'val/')
            val_xml_names = [val_sample_name + '.xml' \
                              for val_sample_name in os.listdir(val_folder) \
                              if os.path.isdir(os.path.join(val_folder, val_sample_name))]

            return train_xml_names, val_xml_names

        train_idxs, val_idxs = train_test_split(range(len(xml_names)),
                                                test_size=val_size)
        train_xml_names = [xml_names[train_idx] for train_idx in train_idxs]
        val_xml_names = [xml_names[val_idx] for val_idx in val_idxs]

        return train_xml_names, val_xml_names

    def convert(self, val_size):
        xml_names = [file_name for file_name in os.listdir(self._xml_label_dir) \
                      if os.path.isfile(os.path.join(self._xml_label_dir, file_name)) and \
                      file_name.endswith('.xml')]
        folders = [file_name for file_name in os.listdir(self._xml_dir) \
                   if os.path.isdir(os.path.join(self._xml_dir, file_name))]
        train_xml_names, val_xml_names = self._train_test_split(folders, xml_names, val_size)

        self._make_train_val_dir()

        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme xml file and save them under images folder
        for target_dir, xml_names in zip(('train/', 'val/'),
                                          (train_xml_names, val_xml_names)):
            for xml_name in xml_names:
                xml_path = os.path.join(self._xml_label_dir, xml_name)
                try:
                    xmldoc = minidom.parse(xml_path)

                    print('Converting %s for %s ...' % (xml_name, target_dir.replace('/', '')))

                    img_path = self._save_yolo_image(xml_name,
                                                     self._image_dir_path,
                                                     target_dir)

                    yolo_obj_list = self._get_yolo_object_list(xmldoc, img_path)
                    self._save_yolo_label(xml_name,
                                          self._label_dir_path,
                                          target_dir,
                                          yolo_obj_list)
                except Exception:
                    print("Convert Error Filename: ", xml_path)

        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()

    def convert_one(self, xml_name):
        xml_path = os.path.join(self._xml_dir, xml_name)
        xml_data = xml.load(open(xml_path))

        print('Converting %s ...' % xml_name)

        img_path = self._save_yolo_image(xml_data, xml_name,
                                         self._xml_dir, '')

        yolo_obj_list = self._get_yolo_object_list(xml_data, img_path)
        self._save_yolo_label(xml_name, self._xml_dir,
                              '', yolo_obj_list)

    def _get_yolo_object_list(self, xmldoc, img_path):
        yolo_obj_list = []

        img_h, img_w, _ = cv2.imread(img_path).shape
        itemlist = xmldoc.getElementsByTagName('object')
        # size = xmldoc.getElementsByTagName('size')[0]
        # width = int((size.getElementsByTagName('width')[0]).firstChild.data)
        # height = int((size.getElementsByTagName('height')[0]).firstChild.data)
        for item in itemlist:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point

            yolo_obj = self._get_other_shape_yolo_object(item, img_h, img_w)

            yolo_obj_list.append(yolo_obj)

        return yolo_obj_list

    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        obj_center_x, obj_center_y = shape['points'][0]

        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 +
                           (obj_center_y - shape['points'][1][1]) ** 2)
        obj_w = 2 * radius
        obj_h = 2 * radius

        yolo_center_x = round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        label_id = self._label_id_map[shape['label']]

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, item, img_h, img_w):
        def __get_object_desc(item):
            __get_dist = lambda int_list: max(int_list) - min(int_list)

            xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
            ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
            xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
            ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data


            x_lists = [int(xmin), int(xmax)]
            y_lists = [int(ymin), int(ymax)]

            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)

        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(item)

        yolo_center_x = round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        classid = (item.getElementsByTagName('name')[0]).firstChild.data
        tmp_label = classid.split('_')

        label_id = self._label_id_map[tmp_label[1]]

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _save_yolo_label(self, xml_name, label_dir_path, target_dir, yolo_obj_list):
        txt_path = os.path.join(label_dir_path,
                                target_dir,
                                xml_name.replace('.xml', '.txt'))

        with open(txt_path, 'w+') as f:
            for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                yolo_obj_line = '%s %s %s %s %s\n' % yolo_obj \
                    if yolo_obj_idx + 1 != len(yolo_obj_list) else \
                    '%s %s %s %s %s' % yolo_obj
                f.write(yolo_obj_line)

    def _save_yolo_image(self, xml_name, image_dir_path, target_dir):
        img_name = xml_name.replace('.xml', '.jpg')
        img_path = os.path.join(image_dir_path, target_dir, img_name)

        if not os.path.exists(img_path):
            source_image_path = os.path.join(self._xml_image_dir, img_name)
            copyfile(source_image_path, img_path)

        return img_path

    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._xml_dir, 'YOLODataset/', 'dataset.yaml')

        with open(yaml_path, 'w+') as yaml_file:
            yaml_file.write('train: %s\n' % \
                            os.path.join(self._image_dir_path, 'train/'))
            yaml_file.write('val: %s\n\n' % \
                            os.path.join(self._image_dir_path, 'val/'))
            yaml_file.write('nc: %i\n\n' % len(self._label_id_map))

            names_str = ''
            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(', ')
            yaml_file.write('names: [%s]' % names_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_dir', type=str, default="./",
                        help='Please input the path of the labelme xml files.')
    parser.add_argument('--val_size', type=float, nargs='?', default=0.2,
                        help='Please input the validation dataset size, for example 0.1 ')
    parser.add_argument('--xml_name', type=str, nargs='?', default=None,
                        help='If you put xml name, it would convert only one xml file to YOLO.')
    args = parser.parse_args(sys.argv[1:])

    convertor = Xml2YOLO(args.xml_dir)
    if args.xml_name is None:
        convertor.convert(val_size=args.val_size)
    else:
        convertor.convert_one(args.xml_name)