# -*- coding: utf-8 -*-
"""
Created on Sun May  1 02:52:05 2022

@author: HINA
"""

"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""
import datetime
import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML, Video
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import base64

from moviepy.editor import *
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
import imutils
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='template')

def to_gray(frame):
    findLaneLines = FindLaneLines()
    gray_frame = findLaneLines.process_image(frame)

    ret, buffer = cv2.imencode(".jpg", gray_frame)
    gray_frame_bytes = buffer.tobytes()
    gray_frame_bytes = base64.b64encode(gray_frame_bytes)
    return ("data:image/jpeg;base64," + str(gray_frame_bytes.decode('utf-8')))

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """

    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)
        resized_out_image = cv2.resize(out_img, (img.shape[1], img.shape[0]))
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, img):
        img = mpimg.imread(img)
        dim_change = cv2.resize(img, (1280, 720))
        out_img = self.forward(dim_change)
        return out_img



    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.preview()
        #out_clip.ipython_display(width=280, height=320)
        #out_clip.write_videofile(output_path, audio=False)
        #out_clip.save_frame("Output")

        """vs = cv2.VideoCapture("output.mp4")
        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
            frame = imutils.resize(frame, width=1200)
            cv2.imshow("Output", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, break from the loop
            if key == ord("q"):
                break"""

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video', methods=['POST','GET'])
def video():
    if('file' in request.files.keys()):
        image = request.files['file']
        return "<img src='"+to_gray(image)+"'  />"
    else:
    	return "200"




@app.route('/img_process', methods=['POST'])
def img_process():
    if ('file' in request.files.keys()):
        image = request.files['file']
        to_gray(image)
        return '{"imgdata":"' + to_gray(image) + '"}'
    else:
        return '{"imgdata":"null"}'


@app.route('/test', methods=['GET'])
def test():
    return "200"


if __name__ == "__main__":
    app.run("192.168.209.96")
#     # app.run(debug=True)"192.168.18.21"
#     # app.run(host="192.168.137.1")

# def main():
#     findLaneLines = FindLaneLines()
#     findLaneLines.process_video("challenge_video.mp4", "output.mp4")
# if __name__ == "__main__":
#     main()
