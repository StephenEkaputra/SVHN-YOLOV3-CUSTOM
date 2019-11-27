import numpy as np
import argparse
import time
import cv2
import os
from glob import glob
import json


labelsPath = os.path.sep.join(['yolo-test', "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join(['yolo-test', "yolo-obj_best.weights"])
configPath = os.path.sep.join(['yolo-test', "yolo-obj.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

dicts = []
diction = {}

num=0
iteration = 0

### SORTING NAME FILE ###
imgs = []
imgs_ready = []
for img_file in os.listdir('images'):
        if img_file.endswith('.png'):
                imgs.append(img_file)
imgs = sorted(imgs,key=lambda x: int(os.path.splitext(x)[0]))
for d in range(len(imgs)):
        imgs_ready.append('images/%s'%imgs[d])

### DETECTION OF IMAGES ###

for fn in imgs_ready:
        image = cv2.imread(fn)
        iteration+= 1
        print(iteration)

        (H, W) = image.shape[:2]

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 256),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []
        boxes_for_json = []
        confidences_for_json = []
        classIDs_for_json = []
        
        for output in layerOutputs:
                for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        if confidence > 0.1:
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
                                boxes_for_json.append([y,x,(y+int(height)),(x+(int(width)))])
                                confidences_for_json.append(float(confidence))
                                classIDs_for_json.append(int(classID))
                                
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.1,
                0.1)

        if len(idxs) > 0:
                for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

        diction = {"bbox" : boxes_for_json, "score" : confidences_for_json, "label" : classIDs_for_json}
        dicts.append(diction)
        #print(dicts)
        
        ###### SAVE THE RESULT IMAGES ######
        num+= 1
        cv2.imwrite("results/%d.png" % num, image)

# SAVE TO JSON FILE

with open('0880817.json', 'w') as fp:
        json.dump(dicts, fp)
