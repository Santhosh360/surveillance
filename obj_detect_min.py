import os
import pathlib
import cv2

import numpy as np
import sys
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_name):
  model_dir = pathlib.Path(model_name)/"saved_model"
  model = tf.saved_model.load(str(model_dir))
  return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
model_name = 'faster_rcnn_resnet101_coco_2018_01_28'
detection_model = load_model(model_name)

def run_inference_for_single_image(model, image):
  if type(image) != np.ndarray:
    image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
 
  return output_dict

def show_inference_video(model,video_path):
  # define a video capture object 
  vid = cv2.VideoCapture(video_path) 
  if (vid.isOpened()== False):
    print("Error opening video stream or file")
    exit(0)
  frame_no = 0
  fps = vid.get(cv2.CAP_PROP_FPS)
  while(True):

    # Capture the video frame 
    # by frame 
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = vid.read() 
    if isinstance(frame, type(None)):
      exit(0)
    frame_no = frame_no + fps
    h, w, c = frame.shape
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Actual detection.
    #frame = cv2.resize(frame, (int(w/2),int(h/2)))
    output_dict = run_inference_for_single_image(model, frame)
    
    for i in range(0,len(output_dict['detection_classes'])):
      if output_dict['detection_classes'][i] != 1:
        output_dict['detection_scores'][i] = 0.0
    #print(output_dict['detection_boxes'])
    #print(output_dict['detection_classes'])
    #print(output_dict['detection_scores'])
    
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        #instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.4)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the resulting frame
    if h > 700:
      frame = cv2.resize(frame, (int(w/2),int(h/2)))
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1000) & 0xFF == ord('q'): 
      break
  
  # After the loop release the cap object 
  vid.release() 
  # Destroy all the windows 
  cv2.destroyAllWindows() 

show_inference_video(detection_model, 'chainSnatching.mkv')
