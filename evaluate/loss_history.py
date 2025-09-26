import numpy as np
import matplotlib.pyplot as plt
f = np.load("/home/alvin/hyper-surrogate-model/saved_model/v1/loss_history.npy")
plt.plot(f[600:])
plt.xlabel("steps")
plt.ylabel("Loss")
plt.xticks(np.arange(600, len(f), step=1000))
plt.xlim(600, len(f))
plt.ylim(0,100)
plt.title("Training Loss History")
plt.grid() 
plt.savefig("/home/alvin/hyper-surrogate-model/saved_model/v1/loss_history.png")