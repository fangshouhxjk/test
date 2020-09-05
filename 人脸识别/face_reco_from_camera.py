# 摄像头实时人脸识别
import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import pandas as pd # 数据处理的库 Pandas
import os           # 读写文件
import studata
import mysqlconn as conn

# 人脸识别模型，提取 128D 的特征矢量
facerec = dlib.face_recognition_model_v1\
    ("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# 计算两个向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("两个向量之间的欧式距离：", dist)
    if dist < 0.4:
        return "same"


# 处理存放所有人脸特征的 CSV
path_features_known_csv = "data/csv_feature_all/" + \
                          studata.class_list[studata.class_cnt]+"/" + \
                          studata.class_list[studata.class_cnt]+"_features_all.csv"
csv_rd = pd.read_csv(path_features_known_csv, header=None)
# 处理存放所有人脸图片的文件夹名称
path_make_dir = "data/data_faces_from_camera/"+\
                studata.class_list[studata.class_cnt]+"/"


path_make_dir_arr = []
for dirs in (os.listdir(path_make_dir)):
    path_make_dir_arr.append(dirs)

features_known_arr = []# 定义一个用来存放所有录入人脸特征的数组
for i in range(csv_rd.shape[0]):# 读取已知人脸数据
    features_someone_arr = []
    for j in range(0, len(csv_rd.ix[i, :])):
        features_someone_arr.append(csv_rd.ix[i, :][j])
    #    print(features_someone_arr)
    features_known_arr.append(features_someone_arr)
print("Faces in Database：", len(features_known_arr))

# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# 创建 cv2 摄像头对象
cap = cv2.VideoCapture(0)
cap.set(3, 480)# cap.set(propId, value)设置视频参数，propId 设置的视频参数，value 设置的参数值

# 返回一张图像多张人脸的 128D 特征
def get_128d_features(img_gray):
    faces = detector(img_gray, 1)
    if len(faces) != 0:
        face_des = []
        for i in range(len(faces)):
            shape = predictor(img_gray, faces[i])
            face_des.append(facerec.compute_face_descriptor(img_gray, shape))
    else:
        face_des = []
    return face_des

# cap.isOpened() 返回 true/false 检查初始化是否成功
while cap.isOpened():
    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)# 取灰度
    faces = detector(img_gray, 0) # 人脸数 faces
    font = cv2.FONT_HERSHEY_COMPLEX# 待会要写的字体
    cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
    pos_namelist = []# 存储所有人脸的名字位置
    name_namelist = []# 存储所有人脸的名字
    if kk == ord('q'):# 按下 q 键退出
        break
    else:
        # 检测到人脸
        if len(faces) != 0:
            # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
            features_cap_arr = []
            for i in range(len(faces)):
                shape = predictor(img_rd, faces[i])
                features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))

            # 遍历捕获到的图像中所有的人脸
            for k in range(len(faces)):
                # 让人名跟随在矩形框的下方
                # 确定人名的位置坐标
                # 先默认所有人不认识，是 unknown
                name_namelist.append("unknown")
                # 每个捕获人脸的名字坐标
                pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                # 对于某张人脸，遍历所有存储的人脸特征
                for i in range(len(features_known_arr)):
                    print("with ",studata.class_list[studata.class_cnt]+"_"+str(path_make_dir_arr[i]), "the ", end='')
                    # 将某张人脸与存储的所有人脸数据进行比对
                    compare = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                    if compare == "same":  # 找到了相似脸
                        if(len(path_make_dir_arr) > i):
                            stunum = path_make_dir_arr[i]
                            stucount =conn.insertdata(stunum)
                            name_namelist[k] = studata.class_list[studata.class_cnt] + "_"  + str(stunum)
                for kk, d in enumerate(faces): # 矩形框
                    # 绘制矩形框
                    cv2.rectangle(img_rd, tuple([d.left(), d.top()]),
                                  tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
            for i in range(len(faces)):# 在人脸框下面写人脸名字
                cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    print("当前识别出的学生班级学号为:", name_namelist, "\n")
    cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("camera", img_rd)# 窗口显示
cap.release()# 释放摄像头
cv2.destroyAllWindows()# 删除建立的窗口
