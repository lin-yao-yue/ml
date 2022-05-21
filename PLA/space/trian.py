import numpy as np
from space.model import Model
from space.dataLoad import load_data
from space.drawFig import Draw

# (98, 1)
data, label = load_data()

print(data)

model = Model()
model.train(data, label)
w = model.w

draw_model = Draw()
draw_model.draw(data, label, w)

