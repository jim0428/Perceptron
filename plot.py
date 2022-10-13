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

    def three_dimension_plot(self):
        f = Figure(figsize=(5, 4), dpi=100)
        f_plot = f.add_subplot(111,projection='3d')
        f_plot.clear()
        zero_class_x_feature = np.array([feature[0] for feature in self.dataset if feature[3] != 0])
        zero_class_y_feature = np.array([feature[1] for feature in self.dataset if feature[3] != 0])
        zero_class_z_feature = np.array([feature[2] for feature in self.dataset if feature[3] != 0])

        first_class_x_feature = np.array([feature[0] for feature in self.dataset if feature[3] != 1])
        first_class_y_feature = np.array([feature[1] for feature in self.dataset if feature[3] != 1])
        first_class_z_feature = np.array([feature[2] for feature in self.dataset if feature[3] != 1])


        f_plot.scatter(zero_class_x_feature,zero_class_y_feature,zero_class_z_feature, color = 'hotpink')
        f_plot.scatter(first_class_x_feature,first_class_y_feature,first_class_z_feature, color = '#88c999')

        x = np.linspace(0, 2, 2)
        y = np.linspace(0, 2, 2)
        X, Y = np.meshgrid(x, y)

        f_plot.plot_surface(
            X,
            Y = -(self.weight[0] / self.weight[2]) - (self.weight[1] / self.weight[2]) - (self.weight[3] / self.weight[2]),
            Z = Y,
            color='g',
            alpha=0.6
        ) 

        f_plot.set_xlabel('X')
        f_plot.set_ylabel('Y')
        f_plot.set_zlabel('Z')

        # x_min = min(min(zero_class_x_feature),min(first_class_x_feature))
        # x_max = max(max(zero_class_x_feature),max(first_class_x_feature))
        # print(x_min,x_max)
        # x = np.arange(x_min - 3,x_max + 3,2)
        # y = (-self.weight[1] / self.weight[2]) * x - (self.weight[0] / self.weight[2])
        # f_plot.plot(x,y)

        canvs = FigureCanvasTkAgg(f, self.window)

        canvs.draw()

        canvs.get_tk_widget().place(x=300,y=80)