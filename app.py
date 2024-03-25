import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import helpers as hp

GAIN_PBAS_DB = 0
GAIN_PBANDE_DB = 0
GAIN_PHAUT_DB = 0
K1 = 1
K2 = 1


def dephasage_180(wave):
    return -1 * wave


def gain_db(wave, gain_db):
    return wave * int(10 ** (gain_db / 10))


def egalisateur(wave_passe_bas, wave_passe_bande, wave_passe_haut, wave, t):
    output = wave_passe_bas + wave_passe_bande + wave_passe_haut
    hp.timeplt1(t, wave, t, output, "Égalisateur")


def passe_bas(wave, t):
    temp, delay = filtre_passe_bas(t, wave)
    temp_dephase = dephasage_180(temp)
    temp_gain = gain_db(temp_dephase, GAIN_PBAS_DB)
    return temp_gain * K1, delay


def filtre_passe_bas(t, wave):
    wc = 2 * np.pi * 700

    b1, a1 = signal.butter(2, wc, 'lowpass', analog=True)
    z1, p1, k1 = signal.tf2zpk(b1, a1)

    print(f'Passe-Bas Numérateur {b1}, Dénominateur {a1}')
    print(f'Passe-Bas Zéros {z1}, Pôles {p1}, Gain {k1}')

    mag1, ph1, w1, fig, ax = hp.bodeplot(b1, a1, "Bode Passe-Bas")
    hp.pzmap1(z1, p1, "PZ Passe-Bas")

    temp = signal.lsim((z1, p1, k1), U=wave, T=t)
    hp.timeplt1(t, wave, temp[0], temp[1], 'Signal filtré')

    delay = - np.diff(ph1) / np.diff(w1)
    hp.grpdel1(w1, delay, 'Passe-Bas')

    # H(s) = 19344425 / s^2 + 6220s + 19344424
    return temp[1], delay


def passe_bande(wave, t):
    temp, delay, w1 = filtre_passe_bande(t, wave)
    temp_gain = gain_db(temp, GAIN_PBANDE_DB)
    return temp_gain * K2, delay, w1


def filtre_passe_bande(t, wave):
    wc1 = 2 * np.pi * 1000
    wc2 = 2 * np.pi * 5000

    b1, a1 = signal.butter(2, wc1, 'highpass', analog=True)
    b2, a2 = signal.butter(2, wc2, 'lowpass', analog=True)

    z1, p1, k1 = signal.tf2zpk(b1, a1)
    z2, p2, k2 = signal.tf2zpk(b2, a2)

    print(f'Passe-haut Numérateur {b1}, Dénominateur {a1}')
    print(f'Passe-haut Zéros {z1}, Pôles {p1}, Gain {k1}')

    print(f'Passe-bas Numérateur {b2}, Dénominateur {a2}')
    print(f'Passe-bas Zéros {z2}, Pôles {p2}, Gain {k2}')

    z, p, k = hp.seriestf(z1, p1, k1, z2, p2, k2)
    b, a = signal.zpk2tf(z, p, k)

    print(f'Passe-bande Numérateur {b}, Dénominateur {a}')
    print(f'Passe-bande Zéros {z}, Pôles {p}, Gain {k}')

    mag1, ph1, w1, fig, ax = hp.bodeplot(b, a, "Bode Passe Bande")
    hp.pzmap1(z, p, "PZ Passe Bande")

    temp = signal.lsim((z, p, k), U=wave, T=t)
    hp.timeplt1(t, wave, temp[0], temp[1], 'Signal filtré')

    delay = - np.diff(ph1) / np.diff(w1)
    hp.grpdel1(w1, delay, 'Delais')

    # PHaut: H(s) = s^2 / s^2 + 8885s + 39478417
    # PBas: H(s) = 98696044 / s^2 + 44428s + 986960440
    # PBande: H(s) = 98696044s^2 / s^2 + 53314s + 38963636400000000
    return temp[1], delay, w1


def passe_haut(wave, t):
    temp, delay = filtre_passe_haut(t, wave)
    temp_dephase = dephasage_180(temp)
    temp_gain = gain_db(temp_dephase, GAIN_PHAUT_DB)

    return temp_gain * 1, delay


def filtre_passe_haut(t, wave):
    wc = 2 * np.pi * 7000

    b1, a1 = signal.butter(2, wc, 'highpass', analog=True)
    z1, p1, k1 = signal.tf2zpk(b1, a1)

    print(f'Passe-haut Numérateur {b1}, Dénominateur {a1}')
    print(f'Passe-haut Zéros {z1}, Pôles {p1}, Gain {k1}')

    mag1, ph1, w1, fig, ax = hp.bodeplot(b1, a1, "Bode Passe-Haut")
    hp.pzmap1(z1, p1, "PZ Passe-Haut")

    temp = signal.lsim((z1, p1, k1), U=wave, T=t)
    hp.timeplt1(t, wave, temp[0], temp[1], 'Signal Sinus 2.5kHz filtré')

    delay = - np.diff(ph1) / np.diff(w1)
    hp.grpdel1(w1, delay, 'Passe Haut')

    # H(s) = s^2 / s^2 + 62200s + 1934442460
    return temp[1], delay


def main():
    #Sinus 2.5khz, 0.25V
    duration = 0.01
    sampling_rate = 100000
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    w = 2 * np.pi * 2500
    wave = 0.25 * np.sin(w * t)

    wave_passe_bas, delay_passe_bas = passe_bas(wave, t)
    #wave_passe_bande, delay_passe_bande, w1 = passe_bande(wave, t)
    #wave_passe_haut, delay_passe_haut = passe_haut(wave, t)

    #delay_egalisateur = delay_passe_haut + delay_passe_bas + delay_passe_bande

    #egalisateur(wave_passe_bas, wave_passe_bande, wave_passe_haut, wave, t)

    #w1 = np.logspace(1, 2000, 5000)
    #hp.timeplt1(t, wave, t, wave_passe_haut, 'Signal Sinus 2.5kHz filtré')
    #hp.grpdel1(w1, delay_egalisateur, 'Passe Haut')
    plt.show()


if __name__ == '__main__':
    main()
