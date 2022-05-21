from multipleLinearRegression.model import Model
from multipleLinearRegression.dataLoad import load_data
import numpy as np
import matplotlib.pyplot as plt

# (450, 14)  (55, 14)
train_data, test_data = load_data()

train_x = train_data[0:450, 0:-1]
train_y = train_data[0:450, -1:]
test_x = test_data[0:55, 0:-1]
test_y = test_data[0:55, -1:]

num_inputs = 13
batch = 450
epochs = 100
# 学习率
lr = 0.01
model = Model(num_inputs)
loss = model.train(train_x, train_y, epochs, lr)
predict_y = model.foward(test_x)
print(model.loss(test_y, predict_y))

plot_x = np.arange(epochs)
# 转换为数组
plot_y = np.array(loss)
plt.plot(plot_x, plot_y)
plt.show()




