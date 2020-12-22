#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import time

import cv2
from openvino.inference_engine import IECore, IENetwork

prob_threshold = 0.4
iou_threshold = 0.4
raw_output_message = False
no_show = False 

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        #print(param)
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        self.isYoloV3 = False

        if param.get('mask'):
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True # Weak way to determine but the only one.

    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [print("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)

def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)

def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects

def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union

def store_video(from_frame,to_frame,cap,out):
    global fps
    for i in range(from_frame,to_frame):
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            out.release()
            exit(0)
        out.write(frame)

def is_motion_detected():
    ret, frame = cap.read()
    if ret == False:
        cap.release()
        out.release()
        exit(0)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    currFrameIndex = int(cap.get(cv2.CAP_PROP_POS_FRAMES));
    static_back = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    static_back = cv2.GaussianBlur(static_back, (25, 25), 0)
    #cv2.imshow("static_back", static_back)
    skipTo = currFrameIndex + fps
    if number_input_frames > skipTo:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skipTo)
    else:
        return None
    check, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    #cv2.imshow("gray", gray)
    diff_frame = cv2.absdiff(static_back, gray)
    #cv2.imshow("diff_frame", diff_frame)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    #cv2.imshow("thresh_frame", thresh_frame)
    cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = 0
    for contour in cnts:
        #print("contour area:", cv2.contourArea(contour))
        if cv2.contourArea(contour) < 100:
            continue
        motion = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        reqy = y-208
        reqx = x-208
        adjy = adjx = 0
        if reqy < 0:
            adjy = reqy
            reqy = 0
        if reqx < 0:
            adjx = reqx
            reqx = 0		
        crop_img = frame[reqy:y+208-adjy, reqx:x+208-adjx]
	#print(reqy-y-208+adjy, reqx-x-208+adjx)
        #cv2.imshow("crop img", crop_img)
        break
    if motion == 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, currFrameIndex)
        return crop_img
    else:
        return None
DEBUG = 0

try:
    if(sys.argv[1]):
        input_stream = sys.argv[1]
except:
    print("input_stream not specified.")

cap = cv2.VideoCapture(input_stream)

if (cap.isOpened() == False):
    print("Error opening video stream or file")
    exit(0)

try:
    with open('coco.names', 'r') as f:
        labels_map = [x.strip() for x in f]
    if DEBUG == 1:
        print(labels_map)
except:
    labels_map = None

ie=IECore()

nw = ie.read_network(model='yolo-v3-tf.xml', weights='yolo-v3-tf.bin')
#nw = ie.read_network(model='frozen_darknet_yolov3_tiny_model.xml', weights='frozen_darknet_yolov3_tiny_model.bin')


avail_dev = ie.available_devices
if 'MYRIAD' in avail_dev:
    print ("MYRIAD is available.")
    curr_dev = 'MYRIAD'
else:
    print("MYRIAD absent, fallback to CPU.")
    curr_dev = 'CPU'
ie.load_network(nw, curr_dev)
print(avail_dev)

# ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
input_blob = next(iter(nw.inputs))
nw.batch_size = 1
n, c, h, w = nw.inputs[input_blob].shape

number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

wait_key_code = 1

# Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
if number_input_frames != 1:
    ret, frame = cap.read()
else:
    wait_key_code = 0

frme = 0
dest_video_path = sys.argv[2]
#print(dest_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
out = cv2.VideoWriter(dest_video_path, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))


# ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
print("Loading model to the plugin")
exec_net = ie.load_network(network=nw, num_requests=2, device_name=curr_dev)

cur_request_id = 0
next_request_id = 1
render_time = 0
parsing_time = 0

# ----------------------------------------------- 6. Doing inference -----------------------------------------------
print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
priority = [0, 1, 2, 3, 5, 7]
nomo = cv2.imread('no_motion_detected.jpg',0)
nopr = cv2.imread('no_priority.jpg',0)
while cap.isOpened():
    priority_in_frame = -1
    frame = is_motion_detected()
    #print(sys.getsizeof(frame))
    if sys.getsizeof(frame) == 16 or sys.getsizeof(frame) == 8:
        print("No motion detected...")
        nomo = cv2.resize(nomo, (w, h))
        cv2.imshow('DetectionResults',nomo)
        cv2.waitKey(1)
        continue

    request_id = cur_request_id
    #print("resize",w,'x',h)
    in_frame = cv2.resize(frame, (w, h))

    # resize input_frame to network size
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))

    # Start inference
    start_time = time()
    exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame})
    det_time = time() - start_time

    # Collecting object detection results
    objects = list()
    if exec_net.requests[cur_request_id].wait(-1) == 0:
        output = exec_net.requests[cur_request_id].outputs
        start_time = time()
        for layer_name, out_blob in output.items():
            out_blob = out_blob.reshape(nw.layers[nw.layers[layer_name].parents[0]].out_data[0].shape)
            layer_params = YoloParams(nw.layers[layer_name].params, out_blob.shape[2])
            if DEBUG == 1:
                print("Layer {} parameters: ".format(layer_name))
                layer_params.log_params()
            objects += parse_yolo_region(out_blob, in_frame.shape[2:],
                                         frame.shape[:-1], layer_params,
                                         prob_threshold)
        parsing_time = time() - start_time

    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    # Drawing objects with respect to the --prob_threshold CLI parameter
    objects = [obj for obj in objects if obj['confidence'] >= prob_threshold]

    if len(objects) and raw_output_message:
        print("\nDetected boxes for batch {}:".format(1))
        print(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

    origin_im_size = frame.shape[:-1]
    for obj in objects:
        if obj['class_id'] in priority:
            #print("class_id",obj['class_id'])
            priority_in_frame = obj['class_id']
        # Validation bbox of detected object
        if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
            continue

        if obj['class_id'] in priority:
            color = (255,255,255)
        else:
            #color = ((obj['class_id'] % 2) * 255, (obj['class_id'] % 3) * 127, (obj['class_id'] % 4 ) * 63)
            color = (0,0,0)
        #print(color,obj['class_id'])
        det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
            str(obj['class_id'])

        if raw_output_message:
            print(
                "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                          obj['ymin'], obj['xmax'], obj['ymax'],
                                                                          color))
        if obj['class_id'] in priority:
            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
            cv2.putText(frame,
                    "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                    (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    currFrameIndex = int(cap.get(cv2.CAP_PROP_POS_FRAMES));
    if priority_in_frame != -1:
        print("Priority content detected:",labels_map[priority_in_frame])
        store_video(currFrameIndex, currFrameIndex + (fps*5), cap, out)
    else:
        print("No priority content detected...")
        nopr = cv2.resize(nopr, (w, h))
        cv2.imshow('DetectionResults',nopr)
        cv2.waitKey(1)
        skipTo = currFrameIndex + fps
        if number_input_frames > skipTo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, currFrameIndex + fps)
        continue
    # Draw performance stats over frame
    inf_time_message = "Inference time: {:.3f} ms".format(det_time * 1e3)
    render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
    parsing_message = "YOLO parsing time is {:.3f} ms".format(parsing_time * 1e3)

    cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
    cv2.putText(frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
    cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

    start_time = time()
    if not no_show:
        cv2.imshow("DetectionResults", frame)
    render_time = time() - start_time

    if not no_show:
        key = cv2.waitKey(wait_key_code)

        # ESC key
        if key == 27:
            break

cv2.destroyAllWindows()

