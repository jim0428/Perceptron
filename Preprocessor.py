from cProfile import label
import math
import random
import numpy as np
from xml.sax.handler import feature_external_ges


class preprocessor:

    def __init__(self):
        pass
    
    #'./NN_HW1_DataSet/NN_HW1_DataSet/基本題/2cring.txt'
    def readfile(self,dataset_url):
        datas = open(dataset_url, 'r')
        dataset = []
        for line in datas:
            stripped_line = line.strip()
            line_list = stripped_line.split()
            line_list = list(map(float,line_list))
            dataset.append(line_list)

        return dataset

    def convert_label(self,dataset):
        check = dict()
        idx = 0
        for pos in range(len(dataset)):
            if dataset[pos][-1] not in check:
                check[dataset[pos][-1]] = idx
                idx += 1
            dataset[pos][-1] = check[dataset[pos][-1]]            
                 
        return dataset

    def split_train_test(self,dataset):
        #先shuffle資料
        random.shuffle(dataset)

        split_boundary = math.ceil(len(dataset) * 2 / 3)

        return dataset[:split_boundary],dataset[split_boundary:]

    def split_feature_label(self,dataset):
        feature = [data[:-1] for data in dataset] 
        label = [data[-1] for data in dataset] 

        return np.array(feature),np.array(label)

