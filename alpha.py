import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import json
import cv2
import six.moves.urllib as urllib
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

# Optimize TensorFlow GPU memory usage
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Disable eager execution for TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

if tf.__version__ < '2.0.0':
    raise ImportError('Please upgrade your TensorFlow installation to v2.0.0 or later!')

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

# Load TensorFlow detection graph
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Initialize label map
sys.path.insert(0, 'utils')
import label_map_util
import visualization_utils as vis_util

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Initialize lists for plotting
time_points = []
person_counts = []
average_waiting_times = []

# Matplotlib setup
plt.ion()
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

def update_plot():
    ax[0].clear()
    ax[1].clear()
    ax[0].plot(time_points, person_counts, marker='o', label='Person Count')
    ax[0].set_title("Person Count Over Time")
    ax[0].set_xlabel("Time (seconds)")
    ax[0].set_ylabel("Person Count")
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(time_points, average_waiting_times, marker='o', label='Avg Waiting Time')
    ax[1].set_title("Average Waiting Time Over Time")
    ax[1].set_xlabel("Time (seconds)")
    ax[1].set_ylabel("Avg Waiting Time (seconds)")
    ax[1].grid(True)
    ax[1].legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

# Video capture
cap = cv2.VideoCapture('queue-test-5.webm')
if not cap.isOpened():
    raise ValueError("Video file cannot be opened. Check the file path or format.")
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0
second_completed = 0

# Person tracking
person_times = {}
existing_data = []

# Detection loop
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            success, image_np = cap.read()
            if not success:
                print('Video ended...')
                break

            frame_count += 1
            if frame_count == (2 * frame_rate):
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

                # Process detections
                detected_boxes = np.squeeze(boxes)
                detected_scores = np.squeeze(scores)
                detected_classes = np.squeeze(classes).astype(np.int32)

                waiting_times = []
                for i in range(detected_boxes.shape[0]):
                    if detected_scores[i] > 0.5 and detected_classes[i] == 1:
                        person_id = f"person_{i}"
                        if person_id not in person_times:
                            person_times[person_id] = {'entry_time': second_completed, 'exit_time': None}
                        person_times[person_id]['exit_time'] = second_completed

                for times in person_times.values():
                    if times['exit_time'] is not None:
                        waiting_times.append(times['exit_time'] - times['entry_time'])

                time_points.append(second_completed)
                person_counts.append(len(person_times))
                average_waiting_times.append(np.mean(waiting_times) if waiting_times else 0)

                update_plot()

                cv2.imshow('Object Detection', image_np)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
