import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import json
import cv2
import six.moves.urllib as urllib
from collections import deque

# Disable eager execution for TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

if tf.__version__ < '2.0.0':
    raise ImportError('Please upgrade your TensorFlow installation to v2.0.0 or later!')

sys.path.insert(0, 'utils')
import label_map_util
import people_class_util as class_utils
import visualization_utils as vis_util

# Model configuration
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'utils/person_label_map.pbtxt'
NUM_CLASSES = 50

# Download and extract model if not already present
if not os.path.exists(MODEL_FILE):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    with tarfile.open(MODEL_FILE) as tar:
        tar.extractall(path=os.getcwd())

# Load model into TensorFlow graph
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Video processing
cap = cv2.VideoCapture('queue-test-5.webm')
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
print(f'Frame Rate: {frame_rate}')
frame_count = 0
second_completed = 0

# Tracking data structures
person_times = {}  # Store the entry and exit times of each person
person_ids = {}  # Store the assigned ID for each detected person

# Read existing data from result.json if available
if os.path.exists("result.json"):
    with open("result.json", "r") as json_file:
        try:
            existing_data = json.load(json_file)
        except json.JSONDecodeError:
            # If JSON is invalid, start with an empty list
            existing_data = []
else:
    existing_data = []

# Open the JSON file in write mode (to overwrite existing file)
with open("result.json", "a") as json_file:
    # Ensure the file starts with an array if empty
    if not existing_data:
        json.dump([], json_file)
        json_file.write("\n")  # Add newline for proper format

    # Detection
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while True:
                success, image_np = cap.read()
                if not success:
                    print('Video is end...')
                    break

                frame_count += 1
                if frame_count == (2 * frame_rate):  # Process every 2 seconds
                    second_completed += 2
                    print(f'Processing at {second_completed} seconds...')
                    frame_count = 0

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
                        line_thickness=8)

                    # Tracking people and waiting time
                    detected_boxes = np.squeeze(boxes)
                    detected_scores = np.squeeze(scores)
                    detected_classes = np.squeeze(classes).astype(np.int32)

                    for i in range(detected_boxes.shape[0]):
                        if detected_scores[i] > 0.5 and detected_classes[i] == 1:  # Person class
                            box = detected_boxes[i]
                            person_id = f"person_{i}"  # Unique ID for each person

                            # Check if person already seen
                            if person_id not in person_times:
                                person_times[person_id] = {'entry_time': second_completed, 'exit_time': None}
                            
                            # Update last seen time
                            person_times[person_id]['exit_time'] = second_completed

                    # Generate annotations
                    annotations = {
                        'person_count': len(person_times),
                        'person_times': {}
                    }

                    # Calculate waiting times
                    for person_id, times in person_times.items():
                        if times['exit_time'] is not None:
                            waiting_time = times['exit_time'] - times['entry_time']
                            annotations['person_times'][person_id] = {
                                'entry_time': times['entry_time'],
                                'exit_time': times['exit_time'],
                                'waiting_time': waiting_time
                            }

                    print(json.dumps(annotations))

                    # Append the new annotation to existing data
                    existing_data.append(annotations)

                    # Write the updated data to the file
                    json.dump(existing_data, json_file, indent=2)  # Ensure the JSON is nicely formatted
                    json_file.write("\n")  # Add newline for proper format

                    cv2.imshow('Object Detection', image_np)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
cap.release()
cv2.destroyAllWindows()
