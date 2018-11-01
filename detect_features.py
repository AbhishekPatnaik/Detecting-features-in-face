import cv2
from imutils.video import VideoStream
import time
import argparse
from imutils import face_utils, translate, resize
import matplotlib.pyplot as plt
import dlib
import numpy as np

plt.ion()

######## Arg Parser ###############
parser = argparse.ArgumentParser()
parser.add_argument("-predictor", required=True, help="path to predictor")
args = parser.parse_args()

print(args.predictor)

vs = VideoStream().start()
time.sleep(1.5)
########## initialize face features
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(args.predictor)

eyelayer = np.zeros((600, 800, 3), dtype=np.uint8)

eyemask = eyelayer.copy()
eyemask = cv2.cvtColor(eyemask, cv2.COLOR_BGR2GRAY)


def detect():
        while(True):
            frame = vs.read()
            frame = resize(frame, width=800)
            eyelayer.fill(0)
            eyemask.fill(0)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            sample = 0
            for rect in rects:
                print("Done loading DLIB library")

                #x,y,w,h = face_utils.rect_to_bb(rect)
                #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 2)

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                lefteye = shape[36:42]
                righteye = shape[42:48]

                cv2.fillPoly(eyemask, [lefteye], 255)
                cv2.fillPoly(eyemask, [righteye], 255)


               # eyelayer = cv2.bitwise_and(frame, frame, mask=eyelayer)
                for points in shape:
                    cv2.circle(frame, tuple(points), 2,(200,0,200))
                    
                plot(rect,sample,frame)    
                
def plot(rect,sample,frame):
        cv2.imwrite('/home/abhishek/data/'+ str(rect) + '.' + str(sample) + '.jpeg', frame)
        plt.imshow(frame)
        plt.pause(0.1)
        print("printing")
        plt.show()

