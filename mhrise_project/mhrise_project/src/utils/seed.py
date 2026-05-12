import numpy as np, random

def seed_all(s=20240117):
    np.random.seed(s); random.seed(s)
