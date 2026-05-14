# show_pkl.py

import pickle


# path = "E:\\项目\\SpoNet-master\\SpoNet-master\\SPO_V4\\results\\PM\\PM_50_[8]\\PM_50_[8]-PM_50_8_'PM50'_20250520T164902-sample1280-t1-0-10000.pkl"  # path='/root/……/aus_openface.pkl   pkl文件所在路径
path = "F:\HRL\HRL-NET\MCLP_1000_30_normal_Normalization.pkl\MCLP_1000_30_normal_Normalization.pkl"  # path='/root/……/aus_openface.pkl   pkl文件所在路径
 # path = "E:\\项目\\SpoNet-master\\SpoNet-master\\SPO_V4\\data\\MCLP\\MCLP_50.pkl"  # path='/root/……/aus_openface.pkl   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)
print(data[0])
print(len(data))
