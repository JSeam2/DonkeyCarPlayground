import cv2
import time
import numpy as np
import hasel
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

    def rgb_to_hsv(self, rgb):
        input_shape = rgb.shape
        rgb = rgb.reshape(-1, 3)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc

        deltac = maxc - minc
        s = deltac / maxc
        deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
        rc = (maxc - r) / deltac
        gc = (maxc - g) / deltac
        bc = (maxc - b) / deltac

        h = 4.0 + gc - rc
        h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
        h[r == maxc] = bc[r == maxc] - gc[r == maxc]
        h[minc == maxc] = 0.0

        h = (h / 6.0) % 1.0
        res = np.dstack([h, s, v])
        return res.reshape(input_shape)

    # def hsv_to_rgb(self, hsv):
    #     """
    #     >>> from colorsys import hsv_to_rgb as hsv_to_rgb_single
    #     >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.60, 0.79, 239))
    #     'r=50 g=126 b=239'
    #     >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.25, 0.35, 200.0))
    #     'r=165 g=200 b=130'
    #     >>> np.set_printoptions(0)
    #     >>> hsv_to_rgb(np.array([[[0.60, 0.79, 239], [0.25, 0.35, 200.0]]]))
    #     array([[[  50.,  126.,  239.],
    #             [ 165.,  200.,  130.]]])
    #     >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.60, 0.0, 239))
    #     'r=239 g=239 b=239'
    #     >>> hsv_to_rgb(np.array([[0.60, 0.79, 239], [0.60, 0.0, 239]]))
    #     array([[  50.,  126.,  239.],
    #            [ 239.,  239.,  239.]])
    #     """
    #     input_shape = hsv.shape
    #     hsv = hsv.reshape(-1, 3)
    #     h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    #     i = np.int32(h * 6.0)
    #     f = (h * 6.0) - i
    #     p = v * (1.0 - s)
    #     q = v * (1.0 - s * f)
    #     t = v * (1.0 - s * (1.0 - f))
    #     i = i % 6

    #     rgb = np.zeros_like(hsv)
    #     v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
    #     rgb[i == 0] = np.hstack([v, t, p])[i == 0]
    #     rgb[i == 1] = np.hstack([q, v, p])[i == 1]
    #     rgb[i == 2] = np.hstack([p, v, t])[i == 2]
    #     rgb[i == 3] = np.hstack([p, q, v])[i == 3]
    #     rgb[i == 4] = np.hstack([t, p, v])[i == 4]
    #     rgb[i == 5] = np.hstack([v, p, q])[i == 5]
    #     rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]

    # return rgb.reshape(input_shape)

    def run_threaded(self):
        # if threaded = True
        # this function is run instead of run

        # You can modify the image frame like any other np array
        #out = self.frame[:][200:600]

        # Test with local photo
        out = Image.open("./mlep.jpg")
        out = np.array(out)


        # Extract reds
        # out_R = out[:,:,0]
        # return out_R

        # convert to HSL
        # out = self.rgb_to_hsv(out)

        # using hasel
        out = hasel.rgb2hsl(out)

        # Using opencv
        #out = cv2.cvtColor(out, cv2.COLOR_RGB2HLS)

        thresh = (0., 1)

        channel = out[:,:,1]

        binary_output = np.zeros_like(channel)
        binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1

        return binary_output




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
