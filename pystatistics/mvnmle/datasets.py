"""
Reference datasets for MVN MLE validation and examples.
EXACT ports of the datasets from R's mvnmle package.
"""

import numpy as np

# Apple dataset - tree size vs worm infestation
# From R: data(apple) - EXACT VALUES
apple = np.array([
    [8.0, 59.0],
    [6.0, 58.0],
    [11.0, 56.0],
    [22.0, 53.0],
    [14.0, 50.0],
    [17.0, 45.0],
    [18.0, 43.0],
    [24.0, 42.0],
    [19.0, 39.0],
    [23.0, 38.0],
    [26.0, 30.0],
    [40.0, 27.0],
    [4.0, np.nan],
    [4.0, np.nan],
    [5.0, np.nan],
    [6.0, np.nan],
    [8.0, np.nan],
    [10.0, np.nan],
])

# Missvals dataset - multivariate data with missing values
# From R: data(missvals) - EXACT VALUES
missvals = np.array([
    [7.0, 26.0, 6.0, 60.0, 78.5],
    [1.0, 29.0, 15.0, 52.0, 74.3],
    [11.0, 56.0, 8.0, 20.0, 104.3],
    [11.0, 31.0, 8.0, 47.0, 87.6],
    [7.0, 52.0, 6.0, 33.0, 95.9],
    [11.0, 55.0, 9.0, 22.0, 109.2],
    [3.0, 71.0, 17.0, np.nan, 102.7],
    [1.0, 31.0, 22.0, np.nan, 72.5],
    [2.0, 54.0, 18.0, np.nan, 93.1],
    [np.nan, np.nan, 4.0, np.nan, 115.9],
    [np.nan, np.nan, 23.0, np.nan, 83.8],
    [np.nan, np.nan, 9.0, np.nan, 113.3],
    [np.nan, np.nan, 8.0, np.nan, 109.4],
])
