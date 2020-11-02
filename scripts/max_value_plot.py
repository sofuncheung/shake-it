import numpy as np
import matplotlib.pyplot as plt


max_value_list = np.load('max_value_list.npy')
plt.plot(max_value_list)
plt.savefig('max_value_list.png', dpi=300)

