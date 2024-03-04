import cv2
import time
import os
import numpy as np
import glob

class vaiCalib():
    def __init__(self):
        super().__init__()

        # Init parameter for AI model
        self.input_image = 'dataset\z3785564845852_002762006e7993079fa5383cc5b4020c.jpg'
        self.output_dir = 'save_dir'
        self.detection_model = 'model/best_dataset.onnx'
        self.segmen_model = 'model/mask_rcnn_dataset_fix.onnx'

        # Input image parameter
        self.color = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        self.width = 640
        self.height = 640
        
        # threshold
        self.scoreThreshold = 0.2
        self.nmsThreshold = 0.4
        self.confidenceThreshold = 0.4
        self.classList = ['obj_1', 'obj_2']
        self.CHECKERBOARD=(13,9)
        self.offex = 20
        self.center_tt = (0,0)

        self.point_Ncontact = 0
        self.indexVthcY = 0
        self.index_maxY = 0
        self.index_nbPcontact = 0

        # image callback function
        self.callback()

    def callback(self):
        # Get path for input image
        imgInput = cv2.imread(os.path.join(os.path.dirname(__file__), self.input_image))
        grayImg = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Find the conners of chess board
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret: 
            print(conners)
            corners = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgPoints = conners.reshape(-1, 2)
            fnl = cv2.drawChessboardCorners(imgInput, (11, 11), corners, ret)
            cv2.imshow("fnl", fnl)
            cv2.waitKey(0)
        else:
            print("No Checkerboard Found")
        
        conners = conners.reshape(-1, 2)


        self.pointOxy = conners[3]