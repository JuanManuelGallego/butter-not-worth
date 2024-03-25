import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import helpers as hp

K1 = 1
K2 = 0.8
GAIN_PBAS_DB = 0
GAIN_PBANDE_DB = 0
GAIN_PHAUT_DB = 0


def gain_db_to_lin(gain_db):
    return int(10 ** (gain_db / 10))


def main():
    # Sinus 2.5khz, 0.25V
    duration = 0.01
    sampling_rate = 100000
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    w = 2 * np.pi * 2500
    wave = 0.25 * np.sin(w * t)

    # Passe bas
    wc_pbas = 2 * np.pi * 700
    b_pbas, a_pbas = signal.butter(2, wc_pbas, 'lowpass', analog=True)
    z_pbas, p_pbas, k_pbas = signal.tf2zpk(b_pbas, a_pbas)

    # Passe haut
    wc_phaut = 2 * np.pi * 7000
    b_phaut, a_phaut = signal.butter(2, wc_phaut, 'highpass', analog=True)
    z_phaut, p_phaut, k_phaut = signal.tf2zpk(b_phaut, a_phaut)

    # Passe Bas et Passe Haut du Passe Bande
    wc1 = 2 * np.pi * 1000
    wc2 = 2 * np.pi * 5000

    b1, a1 = signal.butter(2, wc1, 'highpass', analog=True)
    b2, a2 = signal.butter(2, wc2, 'lowpass', analog=True)

    z1, p1, k1 = signal.tf2zpk(b1, a1)
    z2, p2, k2 = signal.tf2zpk(b2, a2)

    # Passe Bas + Passe Haut
    z_bh, p_bh, k_bh = hp.paratf(z_pbas, p_pbas, k_pbas * -1 * K1 * gain_db_to_lin(GAIN_PBAS_DB), z_phaut, p_phaut, k_phaut * -1 * gain_db_to_lin(GAIN_PHAUT_DB))

    # Passe Bande
    z_bande, p_bande, k_bande = hp.seriestf(z1, p1, k1, z2, p2, k2)

    # Total
    z, p, k = hp.paratf(z_bh, p_bh, k_bh, z_bande, p_bande, k_bande * K2 * gain_db_to_lin(GAIN_PBANDE_DB))
    b, a = signal.zpk2tf(z, p, k)
    mag1, ph1, w1, fig, ax = hp.bodeplot(b, a, "Bode Égaliseur")
    hp.pzmap1(z, p, "Égaliseur")
    temp = signal.lsim((z, p, k), U=wave, T=t)

    delay = - np.diff(ph1) / np.diff(w1)
    hp.grpdel1(w1, delay, 'Delais')
    hp.timeplt1(t, wave, temp[0], temp[1], 'Signal Sinus 2.5kHz filtré')

    print(f'Égaliseur Numérateur {b}, Dénominateur {a}')
    print(f'Égaliseur Zéros {z}, Pôles {p}, Gain {k}')

    plt.show()


if __name__ == '__main__':
    main()
