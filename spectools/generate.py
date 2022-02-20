import numpy as np
from scipy import stats


def new_gauss_peak(spectrum: np.ndarray, counts: int, loc: int, fwhm: float) -> np.ndarray:
    """
    Generates counts in random locations according to a gaussian probability in an existing spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        Count spectrum.
    counts : int
        Number of counts in new peak.
    loc : int
        Location of the new peak.
    fwhm : float
        Full width at half measure of the new peak.

    Returns
    -------
    spectrum : np.ndarray
        Same spectrum as input but with new synthetic peaks added.
    """
    sigma = fwhm / 2.35482004503
    for i in range(counts):
        channel = int(round(stats.norm.rvs(loc, sigma), 0))
        spectrum[channel] += 1
    return spectrum


def new_bg(array, count_per_channel):
    '''
    Generates counts in random locations according to a uniform probability.
    '''
    m = len(array)
    return array + np.random.poisson(count_per_channel, size=m)


def simulated_spectra(iter=100, bg_range=(40, 41), snr_db_range=(5, 6), include_dummy=False, dummy_pos=35,
                      dummy_prob=0.05):
    X = np.zeros((iter, 100))
    y = np.zeros(iter)
    dummy = np.zeros(iter)
    list_peakareas = []
    for i in range(iter):
        snr_db = np.random.randint(snr_db_range[0], snr_db_range[1])
        snr = 10 ** (snr_db / 10.)
        bg = np.random.randint(bg_range[0], bg_range[1])

        peakarea = int(0.5 * (snr ** 2 + np.sqrt(40 * bg * snr ** 2 + snr ** 4)))
        mainpeak = np.random.uniform(0.0, 1.0) <= 0.5
        dummypeak = np.random.uniform(0.0, 1.0) <= dummy_prob
        x = np.zeros(100)
        x = new_bg(x, bg)
        if mainpeak:
            new_gauss_peak(x, counts=peakarea, loc=50, fwhm=3)
            y[i] = 1
            list_peakareas.append(peakarea)
        if dummypeak:
            # Cenario A (nao tem dummy peak)
            if include_dummy:
                new_gauss_peak(x, counts=np.random.randint(60, 100), loc=dummy_pos, fwhm=3)
            dummy[i] = 1
        X[i, :] = x
    return X, y, dummy