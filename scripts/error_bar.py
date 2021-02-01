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

sgd_100 = np.array([
4.3302022239107405,
4.998485304347297,
8.226176006177342,
6.098527293209973,
5.812515714922763
])
sgd_300 =([
16.830455178217917,
10.066523182806693,
5.758828205206326,
5.367702184893953,
9.508981881936537
])
sgd_500 = ([
30.860339220312987,
28.527861343769814,
21.937608467658734,
32.93660396231872,
39.00890912333343
])
sgd_700 = ([
43.57778790850022,
53.24746537494388,
42.79766362217168,
40.31249835432027,
59.97516484497951
])
sgd_900 = ([
70.78653779119402,
71.41533358724125,
74.60602683603629,
72.36607298106658,
56.4773159017397
])

mean_100_sgd = np.mean(sgd_100)
std_100_sgd = np.std(sgd_100)
mean_300_sgd = np.mean(sgd_300)
std_300_sgd = np.std(sgd_300)
mean_500_sgd = np.mean(sgd_500)
std_500_sgd = np.std(sgd_500)
mean_700_sgd = np.mean(sgd_700)
std_700_sgd = np.std(sgd_700)
mean_900_sgd = np.mean(sgd_900)
std_900_sgd = np.std(sgd_900)


x = [100,300,500,700,900]
y_lbfgsb = [mean_100,mean_300,mean_500,mean_700,mean_900]
yerr_lbfgsb = [std_100,std_300,std_500,std_700,std_900]
y_sgd = [mean_100_sgd,mean_300_sgd,mean_500_sgd,mean_700_sgd,mean_900_sgd]
yerr_sgd = [std_100_sgd,std_300_sgd,std_500_sgd,std_700_sgd,std_900_sgd]

fig, ax = plt.subplots()
ax.errorbar(x, y_lbfgsb, yerr=yerr_lbfgsb, marker='s',capsize=5,label='L-BFGS-B 10 iters')
ax.errorbar(x, y_sgd, yerr=yerr_sgd, marker='o', capsize=5, label='SGD 100 epochs')

ax.set_xlabel('Batch Size', fontsize=20)
ax.set_ylabel('Sharpness', fontsize=20)
ax.tick_params(direction='in')
ax.legend()

fig.savefig('sharpness-batch_size.png', dpi=300)


