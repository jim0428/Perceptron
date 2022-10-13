import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ploter:
    def __init__(self,window,weight,dataset):
        self.window = window
        self.weight = weight
        self.dataset = dataset

    def two_dimension_plot(self):
        f = Figure(figsize=(5, 4), dpi=100)
        f_plot = f.add_subplot(111)
        f_plot.clear()
        zero_class_x_feature = np.array([feature[0] for feature in self.dataset if feature[2] != 0])
        zero_class_y_feature = np.array([feature[1] for feature in self.dataset if feature[2] != 0])

        first_class_x_feature = np.array([feature[0] for feature in self.dataset if feature[2] != 1])
        first_class_y_feature = np.array([feature[1] for feature in self.dataset if feature[2] != 1])

        f_plot.scatter(zero_class_x_feature,zero_class_y_feature, color = 'hotpink')
        f_plot.scatter(first_class_x_feature,first_class_y_feature, color = '#88c999')

        x_min = min(min(zero_class_x_feature),min(first_class_x_feature))
        x_max = max(max(zero_class_x_feature),max(first_class_x_feature))
        print(x_min,x_max)
        x = np.arange(x_min - 3,x_max + 3,2)
        y = (-self.weight[1] / self.weight[2]) * x - (self.weight[0] / self.weight[2])
        f_plot.plot(x,y)

        canvs = FigureCanvasTkAgg(f, self.window)

        canvs.draw()

        canvs.get_tk_widget().place(x=300,y=80)