#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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
                current_key = str(words[VALUE_POSITION])
                if current_key not in results:
                    results[current_key] = {'oa': [], 'aa': [], 'kappa': []}
            elif 'OVERALL ACCURACY' in line:
                words = line.split(' ')
                results[current_key]['oa'].append(float(words[VALUE_POSITION]))
            # Check for AA
            elif 'AVERAGE ACCURACY' in line:
                words = line.split(' ')
                results[current_key]['aa'].append(float(words[VALUE_POSITION]))
            # Check for kappa
            elif 'KAPPA COEFFICIENT' in line:
                words = line.split(' ')
                results[current_key]['kappa'].append(float(words[VALUE_POSITION]))

            # Get next line
            line = file.readline()

    for key in results:
        assert len(results[key]['oa']) == len(results[key]['kappa']), 'Wrong list lengths! [1]'
        assert len(results[key]['aa']) == len(results[key]['kappa']), 'Wrong list lengths! [2]'

    return results


# Main for running script independently
def main():
    for data in DATASETS:
        for net in NETWORKS:
            for case in TEST_CASES:
                for noise in NOISE_TYPE:
                    file = 'noise_' + noise + '.nst'
                    path = PATH + net + '/' + data + '/' + case + '/'
                    filename = path + file
                    result = get_values(filename)

                    # print(f'TEST: {net} with {data}')
                    # print('#' * 15)
                    # print(f'OA: {oa.mean():.6f} (+- {oa.std():.6f})')
                    # print(f'AA: {aa.mean():.6f} (+- {aa.std():.6f})')
                    # print(f'Kappa: {kappa.mean():.6f} (+- {kappa.std():.6f})')
                    # print('-' * 15)
                    # print(f'Max OA: {np.max(oa):.5f}')
                    # print(f'Min OA: {np.min(oa):.5f}')
                    # print('')


if __name__ == '__main__':
    main()
