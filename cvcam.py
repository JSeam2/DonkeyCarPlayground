import cv2
import time
import numpy as np

class CvCam(object):
    def __init__(self, iCam=0):
        self.cap = cv2.VideoCapture(iCam)
        self.frame = None
        self.running = True

    def poll(self):
        ret, self.frame = self.cap.read()

    def update(self):
        while(self.running):
            self.poll()

    def run_threaded(self):
        # if threaded = True
        # this function is run instead of run

        # You can modify the image frame like any other np array
        out = self.frame[:][200:600]
        out_R = out[:,:,0]
        return out_R

    def run(self):
        self.poll()
        return self.frame

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()


class CvImageDisplay(object):
    def run(self, image):
        cv2.imshow('frame', image)
        cv2.waitKey(1)

    def shutdown(self):
        cv2.destroyAllWindows()
