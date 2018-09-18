from donkeycar.vehicle import Vehicle
from cvcam import CvCam, CvImageDisplay

v = Vehicle()

# Cam part
cam = CvCam()

# Need to specify outputs
v.add(cam, outputs=["camera/image"], threaded=True)

# display part
disp = CvImageDisplay()

v.add(disp, inputs=["camera/image"])

v.start()
