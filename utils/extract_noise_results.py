#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Set plot style
plt.style.use('bmh')

############
# Set file #
############
PATH = '../../../Results/noise/'

DATASETS = ['paviau', 'indian_pines', 'salinas']
NETWORKS = ['sdmm', 'dffn', 'vscnn', 'sae3ddrn']
TEST_CASES = ['full', 'reduced_10', 'reduced_05', 'reduced_01']
NOISE_TYPE = ['salt_and_pepper', 'additive_gaussian', 'multiplicative_gaussian',
              'section_mul_gaussian', 'single_section_gaussian']

VALUE_POSITION = 3
TEST_SIZE = 10


# Get test results from text file
def get_values(filename):
    results = {}

    current_key = ''
    with open(filename, 'r') as file:
        line = file.readline()
        while line:
            # Check for OA
            if 'amount' in line:
                words = line.split(' ')
                # current_key = str(words[VALUE_POSITION])
                current_key = ''.join(char for char in str(words[VALUE_POSITION]) if char != '\n')

                if current_key not in results:
                    results[current_key] = {'oa': np.array([]), 'aa': np.array([]), 'kappa': np.array([])}
            elif 'OVERALL ACCURACY' in line:
                words = line.split(' ')
                results[current_key]['oa'] = np.append(results[current_key]['oa'], float(words[VALUE_POSITION]))
            # Check for AA
            elif 'AVERAGE ACCURACY' in line:
                words = line.split(' ')
                results[current_key]['aa'] = np.append(results[current_key]['aa'], float(words[VALUE_POSITION]))
            # Check for kappa
            elif 'KAPPA COEFFICIENT' in line:
                words = line.split(' ')
                results[current_key]['kappa'] = np.append(results[current_key]['kappa'], float(words[VALUE_POSITION]))

            # Get next line
            line = file.readline()

    for key in results:
        assert results[key]['oa'].size == results[key]['kappa'].size, 'Wrong list lengths! [1]'
        assert results[key]['aa'].size == results[key]['kappa'].size, 'Wrong list lengths! [2]'

        for noise in results[key]:
            if results[key][noise].size > TEST_SIZE:
                results[key][noise] = results[key][noise][:TEST_SIZE]
            elif results[key][noise].size < TEST_SIZE:
                raise AssertionError

    return results


# Main for running script independently
def main():
    for data in DATASETS:  # Run for 3 datasets
        for case in TEST_CASES:  # Run for 4 test cases
            zero_noise = {}
            for noise in NOISE_TYPE:  # Run for 5 noise types
                nodes = {}
                for net in NETWORKS:  # All network's results will be in the same graphic
                    file = 'noise_' + noise + '.nst'
                    path = PATH + net + '/' + data + '/' + case + '/'
                    filename = path + file

                    results = get_values(filename)

                    nodes[net] = [(noise, results[noise]['oa'].mean()) for noise in results]

                    if noise == 'salt_and_pepper':
                        zero_noise[net] = nodes[net][0]
                    else:
                        nodes[net].insert(0, zero_noise[net])

                # Plot graphs for the simple noise types
                simple_noise_types = ['salt_and_pepper', 'additive_gaussian', 'multiplicative_gaussian']
                if noise in simple_noise_types:
                    # Generate graph for the current values
                    fig, ax = plt.subplots()
                    fig.suptitle(f'Using {noise} noise')

                    size = 0
                    for key in nodes:
                        labels, values = zip(*nodes[key])
                        labels = [f'{str(100 * float(label))}%' for label in labels]
                        size = len(labels)

                        ax.plot(labels, values, linewidth=2.0, label=f'{key} network')

                    ax.set(xlim=(0, size-1), xticks=np.arange(0, size),
                           ylim=(0.0, 1.1), yticks=np.arange(0.2, 1.1, 0.2))
                    ax.set(xlabel='Amount of noise', ylabel='Average accuracy',
                           title=f'Subplot title')
                    ax.legend()
                    # plt.show()
                    plt.savefig(f'{PATH}{data}_{case}_{noise}.png')

                # TODO: Plot graphs for the section noise types
                else:
                    a = 2


if __name__ == '__main__':
    main()
