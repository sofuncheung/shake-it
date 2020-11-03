import numpy as np
import matplotlib.pyplot as plt

train_loss_acc = np.load('train_loss_acc_list.npy')
test_loss_acc = np.load('test_loss_acc_list.npy')

train_loss = train_loss_acc[:,0]
train_acc = train_loss_acc[:,1]
test_loss = test_loss_acc[:,0]
test_acc = test_loss_acc[:,1]

plt.plot(train_loss)
plt.plot(test_loss)
plt.clf()

plt.plot(train_acc)
plt.plot(test_acc)