from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import tarfile
from collections import OrderedDict

sys.path.insert(0, 'utils')
import six.moves.urllib as urllib
import label_map_util
import visualization_utils as vis_util

tf.compat.v1.disable_eager_execution()

app = Flask(__name__)

# Model Configuration
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'utils/person_label_map.pbtxt'
NUM_CLASSES = 50

# Download and Extract Model
if not os.path.exists(MODEL_FILE):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    with tarfile.open(MODEL_FILE) as tar:
        tar.extractall(path=os.getcwd())

# Load TensorFlow Graph
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load Label Map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Thresholds for Anomalies
MAX_WAITING_TIME = 10  # seconds
ENTRY_ZONE = [(0.1, 0.1), (0.5, 0.5)]  # Example restricted entry zone (normalized coordinates)
MIN_BOX_SIZE = 0.01  # Minimum size of a bounding box
MAX_BOX_SIZE = 0.5   # Maximum size of a bounding box


def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    second_completed = 0

    person_times = {}
    total_person_count = 0  # Counter for unique persons detected

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while True:
                success, image_np = cap.read()
                if not success:
                    break

                frame_count += 1
                if frame_count == (2 * frame_rate): 
                    second_completed += 2
                    frame_count = 0

                    # TensorFlow Detection
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # Process Detections
                    detected_boxes = np.squeeze(boxes)
                    detected_scores = np.squeeze(scores)
                    detected_classes = np.squeeze(classes).astype(np.int32)

                    for i in range(detected_boxes.shape[0]):
                        if detected_scores[i] > 0.5 and detected_classes[i] == 1:  # Person class
                            box = detected_boxes[i]
                            ymin, xmin, ymax, xmax = box
                            box_height = ymax - ymin
                            box_width = xmax - xmin
                            person_id = f"person_{len(person_times)}"

                            # Initialize person data if not present
                            if person_id not in person_times:
                                person_times[person_id] = {
                                    'entry_time': second_completed,
                                    'exit_time': None,
                                    'anomalies': []
                                }

                            # Update exit time
                            person_times[person_id]['exit_time'] = second_completed

                            # Anomaly: Entry from Restricted Area
                            if xmin < ENTRY_ZONE[0][0] or xmax > ENTRY_ZONE[1][0]:
                                person_times[person_id]['anomalies'].append(
                                    "Entered from restricted area."
                                )

                            # Anomaly: Abnormal Box Size
                            if box_height * box_width < MIN_BOX_SIZE or box_height * box_width > MAX_BOX_SIZE:
                                person_times[person_id]['anomalies'].append(
                                    "Abnormal size detected."
                                )

    cap.release()

    # Add waiting time anomalies
    for person_id, times in person_times.items():
        if times['exit_time'] is not None:
            waiting_time = times['exit_time'] - times['entry_time']
            if waiting_time > MAX_WAITING_TIME:
                person_times[person_id]['anomalies'].append(
                    f"Waiting too long: {waiting_time} seconds."
                )

    # Sort person_times by numeric ID
    sorted_person_times = OrderedDict(
        sorted(person_times.items(), key=lambda x: int(x[0].split('_')[1]))
    )

    return {
        'total_person_count': len(sorted_person_times),
        'person_data': sorted_person_times
    }


@app.route('/process-video', methods=['POST'])
def process_video_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    video_path = os.path.join('uploads', file.filename)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    file.save(video_path)

    try:
        result = process_video(video_path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
