import numpy as np
import pandas as pd 
import os
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 读取文件并提取特征
def read_files():
    folder_path = './data/CK/'
    label2id = {'anger':0, 'contempt':1, 'disgust': 2,'fear':3,'happy':4,'sadness':5,'surprise':6}
    X = []
    Y = []
    
    # 遍历大文件夹中的每一个文件夹
    for label in os.listdir(folder_path):
        file_path = os.path.join(folder_path, label)
        
        # 遍历小文件夹中的图片
        for file_name in os.listdir(file_path):
            pic_path = os.path.join(file_path, file_name)
            
            # 跳过特定文件路径
            if pic_path == './data/CK/anger\.ipynb_checkpoints':
                continue
            
            # 读取图片并进行预处理
            pic = cv2.imread(pic_path, cv2.IMREAD_COLOR)
            result = cv2.convertScaleAbs(pic)
            
            # 图像归一化
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # 将图片矩阵和标签添加到列表中
            if len(pic):
                X.append(result)
                Y.append(label2id[label])   
    return X, Y

# 提取HOG特征
def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    multichannel = image.shape[-1] > 1
    features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, block_norm='L2-Hys',
                   transform_sqrt=True, feature_vector=True, multichannel=multichannel)
    return features

# 提取特征向量
def features(X):
    x_features = []
    for x in X:
        fd = extract_hog_features(x)
        x_features.append(fd)
    return x_features

# 读取文件并提取特征
X, Y = read_files()

# 提取特征向量
x_features = features(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x_features, Y, test_size=0.2, random_state=42)

# 创建SVM模型并进行训练
svm_model = SVC()
svm_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = svm_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
