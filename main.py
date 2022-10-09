import tkinter as tk


from Preprocessor import preprocessor
from model import perceptron
import numpy as np

def main():
    # 1. 處理資料，先將資料 2/3 當作訓練資料，1/3 當做測試資料
    dataProcessor = preprocessor()
    dataset = dataProcessor.readfile('./NN_HW1_DataSet/NN_HW1_DataSet/基本題/2cring.txt')
    dataset = dataProcessor.convert_label(dataset)
    dataset_train,dataset_test = dataProcessor.split_train_test(dataset)
    dataset_train_feature,dataset_train_label = dataProcessor.split_feature_label(dataset_train)
    dataset_test_feature,dataset_test_label = dataProcessor.split_feature_label(dataset_test)
    
    # 2. 將訓練資料拿去訓練 => 有訓練完的model
    perceptron_ = perceptron(10000, len(dataset_train_feature[0]) ,0.2)
    perceptron_.train(dataset_train_feature,dataset_train_label)
    #print(dataset_train_feature)
    train_predict = perceptron_.test_predict(dataset_train_feature)
    test_predict = perceptron_.test_predict(dataset_test_feature)
    train_score = perceptron_.accuracy(train_predict,dataset_train_label)
    test_score = perceptron_.accuracy(test_predict,dataset_test_label)
    print(train_score,test_score)

    # 3. 將train data 跟 test data 拿去訓練




    # window = tk.Tk()

    # window.geometry("1000x500+200+300")

    # window.title('類神經網路-作業一')
    # job = tk.Label(window, text='Job:')
    # job.grid(row=0, column=0)
    # job_entry = tk.Entry(window)
    # job_entry.grid(row=0, column=1)

    # step = tk.Label(window, text='Step:')
    # step.grid(row=1, column=0)
    # step_entry = tk.Entry(window)
    # step_entry.grid(row=1, column=1)

    # layer_name = tk.Label(window, text='Layer_name:')
    # layer_name.grid(row=2, column=0)
    # layer_entry = tk.Entry(window)
    # layer_entry.grid(row=2, column=1)

    # min_spacing = tk.Label(window, text='min_spacing:')
    # min_spacing.grid(row=3, column=0)
    # min_spacing_entry = tk.Entry(window)
    # min_spacing_entry.grid(row=3, column=1)

    # mybutton = tk.Button(
    #     window, 
    #     text='確認',
    #     # command= lambda: moving_line_and_scrapping_copper(
    #     #     job_entry.get(),
    #     #     step_entry.get(),
    #     #     layer_entry.get(),
    #     #     float(min_spacing_entry.get())
    #     #     )
    # )

    # mybutton.grid(row=4, column=1)
    # window.mainloop()


if __name__ == '__main__':
    main()