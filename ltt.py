from scipy.stats import binom


def bin_p_value(r_hat_times_n, n, alpha):
    return binom.cdf(r_hat_times_n, n, alpha)
