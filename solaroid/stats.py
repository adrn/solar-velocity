import numpy as np


def ln_normal(x, mu, std):
    return -0.5 * (x-mu)**2 / std**2 - 0.5*np.log(2*np.pi) - np.log(std)


def ln_uniform(x, a, b):
    return -np.log(b - a)


def ln_two_sech2(x, h1, h2, f):
    lnterm1 = np.log(f) - 2 * np.log(np.cosh(x / (2 * h1))) - np.log(4 * h1)
    lnterm2 = np.log(1 - f) - 2 * np.log(np.cosh(x / (2 * h2))) - np.log(4 * h2)
    return np.logaddexp(lnterm1, lnterm2)


def ln_exp(x, x0, h):
    return - (x-x0) / h - np.log(h)
