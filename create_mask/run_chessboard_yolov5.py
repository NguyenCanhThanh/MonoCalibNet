
import cv2
import time
import os
import numpy as np
from run_yolo import Yolov5
import glob
# from threading import Thread
# from arduino_cmd import write_data

# cv2.imwrite('ChessboardCalibration_master/camera.jpg',frame)
from processMask import predict_PointA

class Run(Yolov5):
    def __init__(self):
        super().__init__()
        # self.frame = cv2.imread(r'C:\Users\Admin\Desktop\grasp\grasp\frameokie\frame36.jpg')

    def findA(self):
        kc = self.distance(self.disReal,self.pointC)/(1+self.anpha)
        self.oxyC=[self.pointC+[1,0],self.pointC+[0,1]]
        print(self.pointC)
        self.Caculation_Angle(self.oxyC ,self.disReal)
        print(self.angleoxy)
        self.A=self.disReal - np.array([kc if i else -kc for i in self.angleoxy])

    def image(self):
        # print(os.path.join(os.path.dirname(__file__),'anh'))
        for index,k in enumerate(glob.glob('dataset/*.jpg')):
            self.start_time
            self.frame=cv2.imread(k)
            print(index)
            self.get_frame(self.frame)

            # print(self.frame)
            
            self.save_result='resultscasHC'
            self.save_resultYolo='save_reYolo'
            for (classid, confidence, box) in zip(self.result_class_ids, self.result_confidences, self.result_boxes):

                    color = self.colors[int(classid) % len(self.colors)]
                    print('h')
                    try:
                    # np.save(f'{self.save_resultYolo}/{os.path.basename(k)[:-4]}npy',(self.disReal[0],self.disReal[1]))
                      box_seg = self.frame[box[1]-20:box[3]+box[1]+20,box[0]-20:box[0]+box[2]+20]
                      self.frame[box[1]-20:box[3]+box[1]+20,box[0]-20:box[0]+box[2]+20]=self.preSeg.call(box_seg)
                      ##mask = self.preSeg.call(box_seg)
                      #cv2.rectangle(self.frame,(box[0]-20,box[1]-20),(box[0]+box[2]+20,box[3]+box[1]+20),(0,255,0),2)
                      cv2.imwrite(f'seg-TB/{os.path.basename(k)[:-4]}.jpg',self.frame)
                    except:continue

                    # print(box)
                    # cv2.imshow('sdf',self.frame[box[1]:box[3]+box[1],box[0]:box[0]+box[2]])
                    # self.findA()
                    # print(self.A)
                    
            self.stop
            
     # def save_npy(self,i):
    #     np.save(f'{self.save_result}/{os.path.basename(i)[:-4]}npy',self.point3d)
    def video(self):
      pass


a=Run()
# # # a.video()
a.image()
