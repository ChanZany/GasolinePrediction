import numpy as np
import matplotlib.pyplot as plt




def plot_his(history, step, title):
    x = [i for i in range(len(history))][::step]
    y = history[::step]
    plt.xlabel("training step")
    plt.ylabel("training Loss")
    plt.title(title)
    plt.plot(x, y)
    plt.savefig(title)
    plt.show()


loss_history4 = np.fromfile("./loss_historys/Solution4.loss")
loss_history3 = np.fromfile("./loss_historys/Solution3.loss")
plot_his(loss_history3, 1, title="Solution3")
plot_his(loss_history4, 1, title="Solution4")
