# Import packages
import os
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
CWD_PATH = os.getcwd()
MODEL_NAME = 'inference_graph3'
_src = "C:/Users/Amin/Desktop/Model/image"
TEST_IMAGE_PATHS = [ os.path.join(_src, 'test.jpg') ]
Image_size=(10,6)
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
NUM_CLASSES = 6
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (in_width,in_height)=image.size
  return np.array(image.getdata()).reshape(
       (in_height,in_width,3)).astype(np.uint8)
       
       
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef() 
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


import tensorflow.compat.v1 as tf
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)            
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
               image_np,
               np.squeeze(boxes),
               np.squeeze(classes).astype(np.int32),
               np.squeeze(scores),
               category_index,
               use_normalized_coordinates=True,
               line_thickness=8,min_score_thresh=0.8)
            classIdName=[category_index.get(i) for i in classes[0]]
            class_name=classIdName[0]
            values_view = class_name.values()
            value_iterator = iter(values_view)
            idCLass = next(value_iterator)
            nameClass = next(value_iterator)
            
             
            if ( scores[0][0] >=0.90):
                print(scores[0][0])
            else :
                print('I can not detect any bottle ! ' )  