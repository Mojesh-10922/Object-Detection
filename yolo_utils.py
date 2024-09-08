import cv2
import numpy as np

def load_yolo():
    net = cv2.dnn.readNet(r"C:\Users\eswar\OneDrive\Desktop\MINI\yolov4.weights", r"C:\Users\eswar\OneDrive\Desktop\MINI\yolov4 (1).cfg")
    with open(r"C:\Users\eswar\OneDrive\Desktop\MINI\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_objects(detections, width, height, classes):
    boxes = []
    confidences = []
    class_ids = []
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def draw_labels(img, label, confidence, x, y, x_plus_w, y_plus_h, color):
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
