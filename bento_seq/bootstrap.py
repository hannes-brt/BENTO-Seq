import numpy as np
from numpy import newaxis as na


def gen_pdf(inc, exc, n_bootstrap_samples=1000, n_grid_points=100, a=1., b=1., r=0.):
    """Generate bootstrap PDF of PSI"""

    grid = np.arange(1. / (2 * n_grid_points), 1., 1. / n_grid_points)
    pinc = inc.size
    pexc = exc.size
    i_inc = np.random.randint(0, pinc, (pinc, n_bootstrap_samples))
    i_exc = np.random.randint(0, pexc, (pexc, n_bootstrap_samples))

    ninc = np.sum(inc[i_inc], axis=0)
    nexc = np.sum(exc[i_exc], axis=0)

    logpdf = (ninc + a - 1)[:, na] * np.log(grid) + (nexc + b - 1)[:, na] * np.log(1 - grid) - \
             (ninc + nexc)[:, na] * np.log(grid * pinc + (1 - grid) * pexc + r)
    logpdf -= logpdf.max(1)[:,na]
    pdf = np.exp(logpdf)
    pdf /= pdf.sum(1)[:,na]
    pdf = pdf.sum(0) / n_bootstrap_samples

    return pdf, grid

def gen_pdf_multisite(reads, n_bootstrap_samples=1000):
    """Generate bootstrap of PSI for multiple site case."""

    p_reads = np.array([x.size for x in reads])
    i_reads = [np.random.randint(0, pp, (pp, n_bootstrap_samples)) for pp in p_reads]

    n_reads = np.vstack([np.sum(x[ii], axis=0) for x, ii in zip(reads, i_reads)])

    alpha = n_reads.astype(np.float64) * p_reads.min() / p_reads[:, na] + 1
    alpha_0 = alpha.sum(0)[na, :]
    psi = alpha / alpha_0
    psi_mean = np.mean(psi, 1)
    psi_sd = np.sqrt(((alpha * (alpha_0 - alpha)) / (alpha_0 ** 2. * (alpha_0 + 1))).mean(1) + np.var(psi, 1))

    return psi_mean, psi_sd
