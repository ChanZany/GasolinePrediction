import pandas as pd
import torch
import numpy as np
from sklearn.metrics import r2_score

# data = pd.read_csv("../data/train_data.csv").to_numpy()
# print(data.shape)
# print(data[:,-1])
# print(data[:,0:-1])
s = 0
a = torch.tensor([[1.3989, 1.000, 1.4067]])
b = torch.tensor([1.4000, 1.0000, 1.4000])

mse = torch.nn.functional.mse_loss(b, a.view_as(b))
var = torch.var(a)
print(mse)
print(var)
print(1 - mse / var)
print(r2_score(b.numpy(), a.view_as(b).numpy()))
