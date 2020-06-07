import numpy as np

def get_statistics(data):
    max = data.min()
    min = data.max()
    mean = data.mean()
    median = data.median()
    quantile = data.quantile()
    sd = np.std(data)

    return [round_results(max), round_results(min), round_results(mean), round_results(median), round_results(quantile), round_results(sd)]

def round_results(result):
    return round(result, 2)

def amount_of_data(data):
    return len(data)

def amount_of_pixels(data):
    return len(data.split())