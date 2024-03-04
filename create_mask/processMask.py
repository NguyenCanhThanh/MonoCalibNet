
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
from onnxruntime import InferenceSession
from preprocess import Compose, PredictConfig
from cfg import config_grasp

class merge_mask(config_grasp):
    def __init__(self,outputs):
        super().__init__()
        self.results=outputs
        self.kernel=np.ones((3,3),np.uint8)
        # self.i=os.path.basename(img_path)
    def find2pointparrel(self):
        self.img1=self.mask_top
        self.img2=self.mask_below
        self.a=np.array(np.where(self.img1==15))[::-1]
        # print(self.img1.shape)
        assert self.a is not None
        self.b=np.array(np.where(self.img2==75))[::-1]
        self.cx=np.array([min(self.a[0])+abs(min(self.a[0])-max(self.a[0]))/2,min(self.a[1])+abs(min(self.a[1])-max(self.a[1]))/2])

        
        # imgr = cv2.circle(imgr,tuple(np.int32(self.cx)), 2, (255,255,0), thickness=5)
        # cv2.imshow("sdf",imgr)
        #     plt.imsave(f'{path_save}/{os.path.basename(i)}',imgr)
        #     np.save(f'{path_save_npy}/{os.path.basename(i)[:-4]}.npy',cx)
            
        self.img2=cv2.bilateralFilter(self.img2,20,100,2)
        self.img1=cv2.bilateralFilter(self.img1,20,100,2)
        self.img12 = self.mask_top + self.mask_below
        self.img12=cv2.bilateralFilter(self.img12,20,100,2)
        self.img12_egs = cv2.Canny(self.img12, 0,10)
        self.edged = cv2.Canny(self.img1, 0,15)
        self.edged2=cv2.Canny(self.img2, 0, 100)
        cv2.imwrite('img12_gauss.png',self.img12)
        cv2.imwrite('img12_egs.png',self.img12_egs)

        self.edged2copy=self.edged2.copy()
        self.edged2copy=cv2.cvtColor(self.edged2copy,cv2.COLOR_GRAY2BGR)
        # self.center_tt=(0,0)
        # cv2.imshow('segdf',self.edged2)
        
        # cv2.waitKey(0)
        # self.point_contact=0
        # self.index_maxX=0
        
        # self.point_Ncontact=0
        # self.indexVthcY=0
        # self.index_maxY=0
        # self.index_nbPcontact=(0,0)
        self.a1=np.array(np.where(self.edged>0))[::-1].T
        self.b1=np.array(np.where(self.edged2>0))[::-1].T
        assert len(self.a1) >20
        if len(self.b1)==0:
            self.b1=[[0,0]]
            return (self.center_tt,self.cx,self.b1[self.point_contact][self.index_maxX],self.b1[self.point_Ncontact][self.index_maxY])
        

        
        self.distance_seg=np.linalg.norm(self.b1-self.a1[:,None],axis=-1)
        
        self.point_contact=np.array(list(set(np.argmax(self.distance_seg <4,axis=-1))))[1:]

        self.point_Ncontact=np.setxor1d(self.point_contact,np.arange(len(self.b1)))
        for i in self.b1[self.point_contact]:
            self.imgr = cv2.circle(self.edged2copy,tuple(np.int32(i)), 2, (255,0,255), thickness=2)
        # for i in self.b1[self.point_Ncontact]:
        #     self.imgr = cv2.circle(self.imgr,tuple(np.int32(i)), 2, (255,0,0), thickness=2)
        # cv2.imshow('segdfss',self.imgr)

        self.vector_pt=(self.b1[self.point_contact]-self.b1[self.point_Ncontact][:,None])



        self.distance_Parallel =np.linalg.norm(self.vector_pt,axis=-1)

        self.index_maxY=np.argmax(np.sort(self.distance_Parallel)[:,1])
        self.index_maxX=np.argsort(self.distance_Parallel)[:,1][self.index_maxY]

        self.valuePointNB=np.sum(abs(self.b1[self.point_contact]-self.b1[self.point_contact][self.index_maxX]),axis=-1)
        # self.index_nbPcontact=self.b1[self.point_contact][np.where((self.valuePointNB<50)&(self.valuePointNB>15)) ]
        self.index_nbPcontact=self.b1[self.point_contact][np.where(self.valuePointNB <100)]
        # self.index_nbPcontact=self.b1[self.point_contact]
        #####
        self.vector_tt=self.b1[self.point_Ncontact][self.index_maxY]-self.index_nbPcontact
        self.point_coincide =self.index_nbPcontact+self.vector_tt[:,None]
        self.distanct_vthc=np.linalg.norm(self.b1[self.point_Ncontact]-self.point_coincide[:,:,None],axis=-1)
        self.indexVthcY=np.argmin(np.sum(np.min(self.distanct_vthc,axis=-1),axis=-1))
        # // Hieu chinh
        self.center_tt=self.cx+(-self.index_nbPcontact[self.indexVthcY]+self.b1[self.point_Ncontact][self.index_maxY])
        #// chua hieu chinh
        # self.center_tt=self.cx+(-self.b1[self.point_contact][self.index_maxX]+self.b1[self.point_Ncontact][self.index_maxY])
        
    
    def run(self,img_orin):
        return self.getMask(img_orin)
           # return (self.center_tt,self.cx,self.b1[self.point_contact][self.index_maxX],self.b1[self.point_Ncontact][self.index_maxY])
    def getMask(self,img_orin):
        # self.index=np.argmax([len(np.where(i>0)[0]) for i in self.results])
        # self.mask=np.where(np.sum(self.results,axis=0) >0,1,0)
        # self.mask_top=np.where(self.results[self.index] >0,1,0).astype('uint8')
        # self.mask_below=self.mask-self.mask_top
        # self.mask_below=np.where(self.mask_below>0,10,0)
        # self.mask_below=cv2.morphologyEx(self.mask_below.astype('uint8'), cv2.MORPH_OPEN, self.kernel)
        self.re=self.results[0].T[0]

        self.id1=self.results[2][np.where(self.re==1)]
        self.id2=self.results[2][np.where(self.re==0)]

        self.mask_below=np.where(np.sum(self.id1,axis=0) >0,75,0).astype('uint8')
        self.mask_top=np.where(np.sum(self.id2,axis=0) >0,15,0).astype('uint8')
        w,h=self.mask_below.shape
        img_zeros_maskB = np.zeros((w,h,3),np.uint8)
        img_zeros_maskT = np.zeros((w,h,3),np.uint8)
        img_zeros_maskB[:,:,0]=52
        img_zeros_maskB[:,:,1]=50
        img_zeros_maskB[:,:,2]=203
        img_zeros_maskT[:,:,2]=50
        img_zeros_maskT[:,:,1]=82
        img_zeros_maskT[:,:,0]=123
        img_orin[self.mask_below > 0] = img_zeros_maskB[self.mask_below > 0]
        img_orin[self.mask_top > 0] = img_zeros_maskT[self.mask_top > 0]
        
        return img_orin
        # cv2.imshow('a',np.where(np.sum(self.id1,axis=0) >0,120,0).astype('uint8'))
        # cv2.imshow('b',np.where(np.sum(self.id2,axis=0) >0,255,0).astype('uint8'))
        # cv2.waitKey(0)
        # print(self.mask_below.shape)

        # else:
        #     print(self.re)
        #     self.mask_below=np.where(self.results[2][np.where(self.re==0)] >0,10,0).astype('uint8')
        #     self.mask_top=np.where(self.results[2][np.where(self.re==1)] >0,1,0).astype('uint8')
        #     print(self.mask_below.shape)
        


        # self.mask_below=cv2.morphologyEx(mask_below.astype('uint8'), cv2.MORPH_OPEN, self.kernel)
# path=glob.glob(f'{path_save}/**/*')
# for i in path:
#     result=np.load(i)
#     merge_mask(result)
class predict_PointA(config_grasp):
    def __init__(self):
        super().__init__()
        # self.img_seg=img
        self.predictor = InferenceSession(self.modelCascadeMaskRCNN)
        # load infer config
        self.infer_config = PredictConfig(self.infer_cfg)
        # load preprocess transforms
        self.transforms = Compose(self.infer_config.preprocess_infos)
    def call(self,img_seg):

        # predict image
        # save_ig='save_seg'
        # if not os.path.exists(save_ig):
        #   os.mkdir(save_ig)
        # for img_path in img_list:
            # print(img_path)
        # len_img=len(os.listdir(save_ig))
        inputs = self.transforms(img_seg)
        # print(inputs)
        inputs_name = [var.name for var in self.predictor.get_inputs()]
        inputs = {k: inputs[k][None, ] for k in inputs_name}
        outputs = self.predictor.run(output_names=None, input_feed=inputs)
        # np.save(f'{save_ig}/{len_img}.npy',outputs)
        print("ONNXRuntime predict: ")
        # print(outputs[0].shape)
        return merge_mask(outputs).run(img_seg)
    

# predict_PointA(cv)