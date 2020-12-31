from pathlib import Path
import torch
import cv2
import os
from facedetector import FaceDetector
from fastai.vision.widgets import *
import fastbook
from fastbook import *

labelColor = {'Correct': (0,255,0), 'Incorrect': (0,0,255)}

#def num_distrib():
#    "Return the number of processes in distributed training (if applicable)."
#    return int(os.environ.get('WORLD_SIZE', 0))
#
#def distrib_barrier():
#    "Place a synchronization barrier in distributed training so that ALL sub-processes in the pytorch process group must arrive here before proceeding."
#    if num_distrib() > 1 and torch.distributed.is_initialized(): torch.distributed.barrier()

#def load_learner(fname, cpu=True):
#    "Load a `Learner` object in `fname`, optionally putting it on the `cpu`"
#    distrib_barrier()
#    res = torch.load(fname, map_location='cpu' if cpu else None)
#    if hasattr(res, 'to_fp32'): res = res.to_fp32()
#    if cpu: res.dls.cpu()
#    return res

def runVideo(outputPath=None):
    path = Path()
    model_inf = load_learner('c:/Users/Visnu Ritesh/Desktop/Mask_Classifier/export.pkl')

    faceDetector = FaceDetector(
        prototype='c:/Users/Visnu Ritesh/Desktop/Mask_Classifier/deploy.prototxt.txt',
        model='c:/Users/Visnu Ritesh/Desktop/Mask_Classifier/res10_300x300_ssd_iter_140000.caffemodel',
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    webcam = cv2.VideoCapture(0) 
    while True:
        (_, frame) = webcam.read()
#        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceDetector.detect(frame)

        for face in faces:
            xStart, yStart, width, height = face
            
            # clamp coordinates that are outside of the image
            xStart, yStart = max(xStart, 0), max(yStart, 0)

            # predict mask label on extracted face
            faceImg = frame[yStart:yStart+height, xStart:xStart+width]
            pred, pred_idx, probs = model_inf.predict(faceImg)

            # draw face frame
            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (126, 65, 64),
                          thickness=2)

            # center text according to the face frame
            textSize = cv2.getTextSize(pred, font, 1, 2)[0]
            textX = xStart + width // 2 - textSize[0] // 2

            # draw prediction label
            cv2.putText(frame,
                        pred,
                        (textX, yStart-20),
                        font, 1, labelColor[pred], 2)

        cv2.imshow('main', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    runVideo()