import numpy as np


def most_frequent(nums):
    unique_values, freq = np.unique(nums, return_counts=True)
    max_index = np.argmax(freq)
    return unique_values[max_index]
