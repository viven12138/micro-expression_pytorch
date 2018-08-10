import cv2
import os
import config


'''把一个avi拆分成jpg放入workspace
'''

WORKPLACE=config.workplace

def make_frames(avi_path):
    #avi_path视频的绝对路径
    u=avi_path.split('\\')
    vc = cv2.VideoCapture(avi_path)
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        cv2.imwrite(WORKPLACE+'\\'+u[-2]+str(c)+ '.jpg', frame)
        if os.path.getsize(WORKPLACE+'\\'+u[-2]+str(c)+ '.jpg')==0:
            os.remove(WORKPLACE+'\\'+u[-2]+str(c)+ '.jpg')
        c = c + 1
        cv2.waitKey(1)
    vc.release()


