import os
import cv2

def find_max(file_path):
    '''获取file_path指向的avi的帧数'''
    video_cap = cv2.VideoCapture(file_path)
    frame_count = 0
    all_frames = []
    while (True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1
    return len(all_frames)

def find_data_max(FILE):
    '''获取FILE(数据集级别)各个种类的视频数据的最大帧数并以字典的形式返回'''
    the_max_frames = {'anger': 0, 'disgust': 0, 'fear': 0, 'happiness': 0, 'sadness': 0}
    for File1 in os.listdir(FILE):
        tem_max = 0
        temfile = os.path.join(FILE, File1)
        for av_name in os.listdir(temfile):
            av_path = os.path.join(temfile, av_name)
            if find_max(av_path) >= tem_max:
                tem_max = find_max(av_path)
        if File1== 'anger':
            the_max_frames['anger'] = tem_max
        elif File1== 'disgust':
            the_max_frames['disgust'] = tem_max
        elif File1== 'fear':
            the_max_frames['fear'] = tem_max
        elif File1== 'happiness':
            the_max_frames['happiness'] = tem_max
        else:
            the_max_frames['sadness'] = tem_max
    return the_max_frames



