import tkinter as tk
from tkinter import filedialog

from Preprocessor import preprocessor
from model import perceptron
import numpy as np

def predict_data(window,learning_rate,iteration,stop_rate,weight_text,Train_score_text,Test_score_text):
    # 1. 處理資料，先將資料 2/3 當作訓練資料，1/3 當做測試資料
    dataProcessor = preprocessor()
    dataset = dataProcessor.readfile(file_url)
    dataset = dataProcessor.convert_label(dataset)
    dataset_train,dataset_test = dataProcessor.split_train_test(dataset)
    dataset_train_feature,dataset_train_label = dataProcessor.split_feature_label(dataset_train)
    dataset_test_feature,dataset_test_label = dataProcessor.split_feature_label(dataset_test)
    
    # 2. 將訓練資料拿去訓練 => 有訓練完的model
    perceptron_ = perceptron(iteration, len(dataset_train_feature[0]) ,learning_rate)
    perceptron_.train(dataset_train_feature,dataset_train_label)

    # 3. 將train data 跟 test data 拿去預測
    train_predict = perceptron_.test_predict(dataset_train_feature)
    test_predict = perceptron_.test_predict(dataset_test_feature)
    train_score = perceptron_.accuracy(train_predict,dataset_train_label)
    test_score = perceptron_.accuracy(test_predict,dataset_test_label)
    print(train_score,test_score)

    weight_text.set(perceptron_.weight) # 設定 weight 的內容
    Train_score_text.set(train_score)   # 設定 train_score 的內容
    Test_score_text.set(test_score)     # 設定 test_score 的內容
    
    tk.Label(window, textvariable=weight_text).place(x=100, y=170)
    
    tk.Label(window, textvariable=Train_score_text).place(x=100, y=200)
    
    tk.Label(window, textvariable=Test_score_text).place(x=100, y=230)

    if(len(dataset_train_feature[0] == 2)):
        #畫2D的圖
        two_dimension_plot(perceptron_.weight,dataset_test_feature,dataset_test_feature)

def two_dimension_plot(weight,dataset_train_feature,dataset_test_feature):
    pass

def get_file_url(file_name):
    global file_url 
    file_url = filedialog.askopenfilename()
    
    file_name.set(file_url.split('/')[-1])
    
    print(file_url)

def main():
    window = tk.Tk()

    window.geometry("1000x500+200+300")

    window.title('類神經網路-作業一')
    #選擇檔案
    file_name = tk.StringVar()   # 設定 text 為文字變數
    file_name.set('')            # 設定 text 的內容

    tk.Label(window, text='請選擇檔案').place(x = 20,y = 20)

    tk.Button(window, text='選擇檔案',command= lambda: get_file_url(file_name)).place(x = 120,y = 16)
    
    tk.Label(window, textvariable=file_name).place(x=180, y=20)

    #學習率

    tk.Label(window, text='Learning rate:').place(x = 20,y = 50)
    learning_rate = tk.Entry(window)
    learning_rate.place(x = 120,y = 50)

    #輸入迭代次數

    tk.Label(window, text='Iteration:').place(x = 20,y = 80)
    interation = tk.Entry(window)
    interation.place(x = 120,y = 80)
   
    
    #Stop rate:
    tk.Label(window, text='Stop rate:').place(x = 20,y = 110)
    stop_rate = tk.Entry(window)
    stop_rate.place(x = 120,y = 110)
    

    #Weight
    tk.Label(window, text='Weight:').place(x = 20,y = 170)
    weight_text = tk.StringVar()         # 設定 weight 為文字變數
    weight_text.set('')                  # 設定 weight 的內容

    #Train score
    tk.Label(window, text='Train score:').place(x = 20,y = 200)
    Train_score_text = tk.StringVar()    # 設定 Train_score 為文字變數
    Train_score_text.set('')             # 設定 Train_score 的內容

    #Test score
    tk.Label(window, text='Test score:').place(x = 20,y = 230)
    Test_score_text = tk.StringVar()     # 設定 Test_score 為文字變數
    Test_score_text.set('')              # 設定 Test_score 的內容


    #開始預測資料
    tk.Button(window, text='確認',command= lambda: predict_data(window,float(learning_rate.get()),int(interation.get()),(stop_rate.get()),weight_text,Train_score_text,Test_score_text)).place(x = 80,y = 140)


    
    window.mainloop()


if __name__ == '__main__':
    main()