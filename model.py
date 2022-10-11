import numpy as np

class perceptron:
    def __init__(self,iteration,weight_len,learning_rate):
        self.iteration = iteration
        self.weight = np.zeros(weight_len + 1) #多一個給bias
        self.learning_rate = learning_rate
    
    def accuracy(self,actual,answer):
        correct = 0.0
        for actual_,answer_ in zip(actual,answer):
            if(actual_ == answer_):
                correct += 1
        
        return correct / float(len(actual)) * 100.0

    def allDataToPredict(self,dataset_feature):
        test_predict = np.array([])
        for feature in dataset_feature:
            test_predict = np.append(test_predict,self.predict(feature))
        
        return test_predict

    def predict(self,feature):
        # v = feature * weight + bias 
        # v > 0 => prediction 1
        # v < 0 => prediction -1

        prediction = np.dot(feature,self.weight[1:]) + self.weight[0]
        #predict_ans = -1
        if prediction > 0:
            predict_ans = 1
        else:
            predict_ans = 0
        return predict_ans

    def train(self,stop_score,dataset_train_feature,dataset_train_label):
        for i in range(self.iteration):
            train_predict = self.allDataToPredict(dataset_train_feature)
            train_score = self.accuracy(train_predict,dataset_train_label)
            if(train_score >= stop_score):
                return i + 1
            for feature,label in zip(dataset_train_feature,dataset_train_label):
                predict_ans = self.predict(feature)
                
                if(predict_ans == 1 and label == 0):
                    # w(n) - nx(n)
                    self.weight[1:] = self.weight[1:] - (self.learning_rate * feature)
                    self.weight[0] = self.weight[0] - (self.learning_rate)
                elif(predict_ans == 0 and label == 1):
                    # w(n) + nx(n)
                    self.weight[1:] = self.weight[1:] + (self.learning_rate * feature)
                    self.weight[0] = self.weight[0] + (self.learning_rate)   
        return self.iteration
