import numpy as np
import matplotlib.pyplot as plt

from config import net_configs, results_dir

sample_labels = {}
sample_results = {}
sample_times = {}
population_results = []
population_times = []

def calc_distribution(sample, name):
    mean = np.mean(sample)
    std = np.std(sample)
    print(f'[{name}] size: {len(sample)} mean: {mean} std: {std}')
    return (mean, std)

def z_test(population, sample, name):
    stderr = np.std(population) / (sample.size**(0.5))
    z = (np.mean(sample) - np.mean(population)) / stderr
    print(f'[{name}] z-value: {z}')

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
    calc_distribution(sample_results[config["name"]], 'result')
    z_test(population_results, sample_results[config["name"]], 'result')
    calc_distribution(sample_times[config["name"]], 'time')
    z_test(population_times, sample_times[config["name"]], 'time')
    print('\n')
    
    plt.subplot(3, 1, i+1)
    plt.title(config["name"])
    #plt.xlim(1000, 4000)
    plt.hist(sample_times[config["name"]], 200)

print('analyzing entire population')
calc_distribution(population_results, 'result')
calc_distribution(population_results, 'time')
plt.show()
