import os
import pathlib
import cv2

import numpy as np
import sys
import tensorflow as tf
import math

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
model_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
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

def is_person_detected(model,start_frame,vid,out,fps):
  person_in_frame = 0
  fixed_frame = start_frame+fps
  print('<<<--------PERSON_DETECTION_START-------->>>')
  while ( start_frame < fixed_frame ):
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = vid.read()
    if ret == False:
      out.release()
      vid.release()
      exit(0)
    output_dict = run_inference_for_single_image(model, frame)
    print('FRAME_START>>>>>>>>')
    print('FRAME_NUMBER:',vid.get(1))
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
    frame,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    line_thickness=5,
    min_score_thresh=0.35)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      out.release()
      vid.release()
      exit(0)
    for j in range(0,len(output_dict['detection_classes'])):
      if output_dict['detection_classes'][j] == 1 and output_dict['detection_scores'][j] > 0.35:
        print("Person detection score in frame",start_frame,':',output_dict['detection_scores'][j])
        person_in_frame = person_in_frame + 1
        break
    if person_in_frame:
      print("person_in_frame count:",person_in_frame)
      print("Person detected!!!")
      print('FRAME_END>>>>>>>>>>')
      print('<<<---------PERSON_DETECTION_END--------->>>')
      return True
    print('Person not detected...')
    print('FRAME_END')
    start_frame+=5
  print('<<<---------PERSON_DETECTION_END--------->>>')
  return False

def store_video(from_frame,to_frame,vid,out):
  global fps
  global person_detected_in_past
  global person_detected_in_present
  if (person_detected_in_past != person_detected_in_present):
    curr_frame = from_frame - fps
    if curr_frame < 0:
      curr_frame = 0
  else:
    curr_frame = from_frame
  prev_frameIndex = vid.get(1)
  vid.set(1,curr_frame)
  for i in range(curr_frame,to_frame):
    ret, frame = vid.read()
    if ret == False:
      vid.release()
      out.release()
      exit(0)
    out.write(frame)
  vid.set(1,prev_frameIndex)

person_detected_in_past = False
frme = 0
video_path = sys.argv[1]
vid = cv2.VideoCapture(video_path)
if (vid.isOpened()== False):
  print("Error opening video stream or file")
  exit(0)
dest_video_path = sys.argv[2]
print(dest_video_path)
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
fps = int(round(vid.get(cv2.CAP_PROP_FPS)))
out = cv2.VideoWriter(dest_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
try:
  while True:
    person_detected_in_present = is_person_detected(detection_model, frme, vid, out, fps)
    if person_detected_in_present == False:
      frme = frme + fps
      person_detected_in_past = False
      print("ff 1sec")
      continue
    print("Person detected:", person_detected_in_present)
    store_video(frme, frme + (fps*5), vid, out)
    person_detected_in_past = person_detected_in_present
    frme = frme + (fps*5)
    print("ff 5secs")
except KeyboardInterrupt:
  print("Ctrl+C Pressed -- Exiting Application!!!")
  vid.release()
  out.release()
except Exception as e:
  print("Unknown Error Occurred -- Exiting Application!!!")
  vid.release()
  out.release()
