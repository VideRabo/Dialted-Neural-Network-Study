import numpy as np
import matplotlib.pyplot as plt
import math

from config import net_configs, results_dir, reference_sample

sample_labels = {}
sample_results = {}
sample_times = {}
population_results = []
population_times = []

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def calc_distribution(sample, name):
    mean = np.mean(sample)
    std = np.std(sample)
    print(f'[{name}] size: {len(sample)} mean: {mean} std: {std}')
    return (mean, std)

def z_test(population, sample, name):
    stderr = np.std(population) / (sample.size**(0.5))
    z = (np.mean(sample) - np.mean(population)) / stderr
    print(f'[{name}] z-score: {z}')

def bernoulli_two_sided_test(reference, sample, name):
    P0 = 0.5 #np.mean(reference)
    N = 50 #len(sample)
    correct = 20 #np.sum(sample)
    P_err_1 = 0

    t = int(N * P0 - correct)
    # print(P0, N, correct, t)
    for K in range(0, t, 1 if t >= 0 else -1):
        P_err_1 += nCr(N, K) * P0**K * (1 - P0)**(N - K)
    
    # print(P_err_1)
    t = int(N * P0 + correct)
    for K in range(t, N, 1 if t < N else -1):
        P_err_1 += P0**K * (1-P0)**(N - K)

    print(f'[{name}] bernoulli - P of type 1 err: {P_err_1}')
    

# load data from file to lists
for config in net_configs:
    print(f'loading {config["name"]}')
    data = np.loadtxt(f'{results_dir}/{config["name"]}')
    sample_labels[config["name"]] = data[:, 0]
    sample_results[config["name"]] = data[0:, 1]
    sample_times[config["name"]] = data[1:, 2]

    population_results.extend(sample_results[config["name"]])
    population_times.extend(sample_times[config["name"]])

# perform statistical operations on samples
for i, config in enumerate(net_configs):  
    print(f'analyzing {config["name"]}')
    
    calc_distribution(sample_results[config["name"]], 'results')
    #bernoulli_two_sided_test(sample_results[reference_sample], sample_results[config["name"]], 'results')


    calc_distribution(sample_times[config["name"]], 'time')
    z_test(sample_times[reference_sample], sample_times[config["name"]], 'time')
    print('\n')
    
    plt.subplot(3, 1, i+1)
    plt.title(config["name"])
    #plt.xlim(1000, 4000)
    plt.hist(sample_times[config["name"]], 200)

print('analyzing entire population')

calc_distribution(population_times, 'time')
plt.tight_layout()
plt.show()
