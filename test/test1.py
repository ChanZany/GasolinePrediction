import pandas as pd

data = pd.read_csv("../data/train_data.csv").to_numpy()
print(data.shape)
print(data[:,-1])
print(data[:,0:-1])
