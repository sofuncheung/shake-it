import numpy as np
import matplotlib.pyplot as plt

bs_100 = np.array([7.583804566413164,
                   8.407078340649605,
                   4.0459686405956745,
                   2.3008040711283684,
                   7.853351511061192])
bs_300 = np.array([32.88629722595215,
                   18.58888142183423,
                   17.153964422643185,
                   24.275504250079393,
                   68.38934242725372])
bs_500 = np.array([139.8961627855897,
                   93.25310738012195,
                   124.29409970156848,
                   36.658488251268864,
                   74.52569440379739])
bs_700 = np.array([162.75092773884535,
                   81.97738244384527,
                   213.6460152156651,
                   239.98156198486686,
                   282.52355014532804])
bs_900 = np.array([226.67552831023932,
                   261.53376215323806,
                   265.2023925818503,
                   163.4903162829578,
                   297.6482810266316])
mean_100 = np.mean(bs_100)
std_100 = np.std(bs_100)
mean_300 = np.mean(bs_300)
std_300 = np.std(bs_300)
mean_500 = np.mean(bs_500)
std_500 = np.std(bs_500)
mean_700 = np.mean(bs_700)
std_700 = np.std(bs_700)
mean_900 = np.mean(bs_900)
std_900 = np.std(bs_900)

'''
sgd_100 =
sgd_300 =
sgd_500 =
sgd_700 =
sgd_900 =
'''

x = [100,300,500,700,900]
y_lbfgsb = [mean_100,mean_300,mean_500,mean_700,mean_900]
yerr_lbfgsb = [std_100,std_300,std_500,std_700,std_900]

plt.errorbar(x, y_lbfgsb, yerr=yerr_lbfgsb, marker='s',capsize=5)

plt.xlabel('Batch Size', fontsize=20)
plt.ylabel('Sharpness', fontsize=20)
plt.grid(which='major')

plt.savefig('sharpness-batch_size.png', dpi=300)


