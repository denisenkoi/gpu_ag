import numpy as np


def check_uniform(array, step):
    for i, value in enumerate(array):
        if i < len(array) - 1:
            next_calculated_depth = value + step
            next_depth = array[i + 1]
            if abs(next_calculated_depth - next_depth) > step * 1e-9:
                return False
    return True
