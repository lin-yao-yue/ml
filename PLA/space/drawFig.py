import matplotlib.pyplot as plt
import numpy as np


class Draw:
    def draw(self, data, label,  w):
        plt.figure()
        x1 = []
        x2 = []
        for i in range(len(label)):
            x1.append(data[i][0])
            x2.append(data[i][1])
            if label[i][0] == 1:
                plt.plot(data[i][0], data[i][1], 'o' + 'r', ms=10)
            else:
                plt.plot(data[i][0], data[i][1], '*' + 'b', ms=10)

        d_y = []
        for i in x1:
            d_y.append(-(w[0][0]*i+w[2][0])/w[1][0])
        plt.plot(x1, d_y)

        plt.show()
