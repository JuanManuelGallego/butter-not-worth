"""
Fichiers d'exemples pour la problématique de l'APP6 (S2)
(c) Julien Rossignol & JB Michaud Université de Sherbrooke
v 1.1 Hiver 2024

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import helpers as hp


###############################################################################
def exampleRacines():
    """
    Calcule et affiche les pôles et zéros d'une FT arbitraire

    :return:
    """
    b1 = [1, 0, 0]  # définition des facteurs du polynome numérateur
    a1 = [1, 1, 0.5]  # définition des facteurs du polynome dénominateur

    # méthode plus manuelle
    z1 = np.roots(b1)
    # pour éventuellement faire l'inverse de l'opération roots, i.e. passer d'une
    # liste de racines aux coefficients du polynôme, voir np.poly
    p1 = np.roots(a1)
    print(f'Racine exemple 1 Zéros:{z1}, Pôles:{p1}')  # affichage du resultat dans la console texte
    # appel d'une fonction pour affichage graphique, voir autre fichier helpers.py
    hp.pzmap1(z1, p1, 'Example de racines 1')

    # méthode utilisant scipy
    (z2, p2, k2) = signal.tf2zpk(b1, a1)  # passage à la représentation zok (liste de racines + "gain")
    print(f'Racine exemple 2 Zéros : {z2}, Pôles: {p2}, Gain: {k2}')
    hp.pzmap1(z2, p2, 'Example de racines 2')


###############################################################################
def exampleBode():
    """
    Calcule et affiche le lieu de bode d'une FT arbitraire

    :return:
    """
    b1 = [1, 0, 0]  # définition du numérateur de la fonction de transfert
    a1 = [1, 1, 0.5]  # définition du dénominateur de la fonction de transfert

    # méthode 1 avec bode
    tf1 = signal.TransferFunction(b1, a1)  # définit la fonction de transfert
    # calcul le diagrame de Bode, la magnitude en dB et la phase en degrés la fréquence en rad/s
    w1, mag1, phlin1 = signal.bode(tf1, np.logspace(-1.5, 1, 200))  # on pourrait aussi laisser la fonction générer les w
    # fonction d'affichage
    hp.bode1(w1, mag1, phlin1, 'Example 1')


###############################################################################
def exampleButterworth():
    """
    Exemple de génération et affichage pour la FT d'un filtre de butterworth d'ordre 4

    :return:
    """

    order = 4
    wn = 1   # frequence de coupure = 1 rad/s
    # définit un filtre passe bas butterworth =>  b1 numerateur, a1 dénominateur
    b1, a1 = signal.butter(order, wn, 'low', analog=True)
    print(f'Butterworth Numérateur {b1}, Dénominateur {a1}')  # affiche les coefficients correspondants au filtre
    print(f'Racine butterworth Zéros:{np.roots(b1)}, Pôles:{np.roots(a1)}')  # affichage du resultat dans la console texte


    # Réponse en fréquence
    mag1, ph1, w1, fig, ax = hp.bodeplot(b1, a1, 'Butterworth Example')

    # Délai de groupe
    delay = - np.diff(ph1) / np.diff(w1)    # calcul
    hp.grpdel1(w1, delay, 'Exemple butterworth')    # affichage

    # Exemple 1 réponse temporelle 3 cos
    t1 = np.linspace(0, 10, 5000, endpoint=False)
    # génération de 3 sinusoîdes
    u1 = (np.cos(2 * np.pi * 0.4 * t1) + 0.6 * np.cos(2 * np.pi * 4 * t1) +
          0.5 * np.cos(2 * np.pi * 8 * t1))
    # simulation temporelle du système
    tout1, yout1, xout1 = signal.lsim((b1, a1), u1, t1)
    hp.timeplt1(t1, u1, tout1, yout1, 'filtrage de 3 sinus')   # affichage

    # Exemple 2 réponse à l'échelon à la mitaine mitaine
    t2 = np.linspace(0, 10, 5000, endpoint=False)
    u2 = np.ones_like(t2)
    tout2, yout2, xout2 = signal.lsim((b1, a1), u2, t2)
    hp.timeplt1(t2, u2, tout2, yout2, 'échelon avec lsim')

    # Exemple 3 réponse à l'échelon via scipy.signal
    tout3, yout3 = signal.step((b1, a1))
    t3m = np.amax(tout3)
    stept3 = t3m/len(tout3)
    hp.timeplt1(np.arange(0, t3m, stept3), np.ones_like(tout3),
                tout3, yout3, 'échelon avec signal.step')


###############################################################################
def probleme1():
    """
    Framework pour le problème 1 du laboratoire

    :return:
    """
    #Question A
    
    ##FT1
    #Entrez les poles, les zeros et le gain de la FT #1
    z1 = [3]
    p1 = [-3]
    k1 = 1
    
    #Ajouter le code pour obtenir la représentation de la fonction de transfert sous forme de polynome. Indice: Regarder la fonction scipy.signal.zpk2tf
    h1 = signal.zpk2tf(z1, p1, k1)
    hp.bodeplot(h1[0], h1[1], "Bode FT1")
    #Ajouter le code pour generer la carte des poles et zeros et le lieu de Bode
    #utiliser les fonctions dans helpers.py
    hp.pzmap1(z1, p1, "FT1")

    ##FT2
    #Entrez les poles, les zeros et le gain de la FT #2
    z2 = [3+5j, 3-5j]
    p2 = [-3+5j, -3-5j]
    k2 = 1
    
    #Ajouter le code pour obtenir la représentation de la fonction de transfert sous forme de polynome
    h2 = signal.zpk2tf(z2, p2, k2)
    hp.bodeplot(h2[0], h2[1], "Bode FT2")

    
    #Ajouter le code pour generer la carte des poles et zeros et le lieu de Bode
    #utiliser les fonctions dans helpers.py
    hp.pzmap1(z2, p2, "FT2")

    #Question B
    
    #Analysez bien les lignes suivantes pour comprendre leur fonctionnement
    # définit les entrées pour la simulation temporelle, i.e. quelques sinusoîdes de fréquence différentes
    t = np.linspace(0, 10, 5000)  # 5000 points de 0 à 10s
    w = [0, 1, 4, 15, 50]   # valeurs des fréquences désirées, rad/s
    # génère une constante si w = 0 sinon un sinus
    u = [np.ones_like(t) if w[i] == 0 else np.sin(w[i] * t) for i in range(len(w))]
    # exemple réponse temporelle pour plusieurs entrées sinusoidales définies ci-dessus
    # initialise les listes qui vont contenir l'information retournée par lsim
    tout = []
    yout1 = []
    yout2 = []
    # itère sur les fréquences désirées
    for i in range(len(w)):
        temp = signal.lsim((z1, p1, k1), U=u[i], T=t)  # temp = [t, y, x], voir l'aide de lsim
        tout.append(temp[0])
        yout1.append(temp[1])
        
        temp = signal.lsim((z2, p2, k2), U=u[i], T=t)  # répète pour l'autre FT
        yout2.append(temp[1])
        
    #Affichage des simulations
    hp.timepltmulti1(t, u, w, tout, yout1, 'H1')
    hp.timepltmulti1(t, u, w, tout, yout2, 'H2')

    #Question C
    #Comment mettre les filtres en serie/parrallele? Reflechisser a deux approches possibles, l'une utilisant lsim et l'autre utilisant les fonctions paratf et seriestf de la librairie helper
    #Ensuite, vous pouvez utiliser le meme genre de boucle que ci-dessous pour generer les reponses aux differentes frequences

    #Serie
    toutSerie = []
    youtSerie = []
    for i in range(len(w)):
        temp1 = signal.lsim((z1, p1, k1), U=u[i], T=t)
        toutSerie.append(temp1[0])

        temp2 = signal.lsim((z2, p2, k2), U=temp1[1], T=t)
        youtSerie.append(temp2[1])

    hp.timepltmulti1(t, u, w, toutSerie, youtSerie, 'Serie')

    #Parallele
    toutPara = []
    youtPara = []
    zPara, pPara, kPara = hp.paratf(z1, p1, k1, z2, p2, k2)

    for i in range(len(w)):
        tempPara = signal.lsim((zPara, pPara, kPara), U=u[i], T=t)
        toutPara.append(tempPara[0])
        youtPara.append(tempPara[1])

    hp.timepltmulti1(t, u, w, toutPara, youtPara, 'Para')

    #Question D
    #Inspirez-vous de la question a pour afficher la carte des poles et zeros et le lieu de Bode de ces filtres, afficher aussi le delais de groupe en utilisant hp.grpdel1 en vous inspirant de la fonction exempleButterworth

    #Para
    b_para, a_para = signal.zpk2tf(zPara, pPara, kPara)
    mag1, ph1, w1, fig, ax = hp.bodeplot(b_para, a_para, "FT Para")
    hp.pzmap1(zPara, pPara, "PZ Para")

    delayPara = - np.diff(ph1) / np.diff(w1)    # calcul
    hp.grpdel1(w1, delayPara, 'Delais Para')

    #Serie
    zSerie, pSerie, kSerie = hp.seriestf(z1, p1, k1, z2, p2, k2)
    b_serie, a_serie = signal.zpk2tf(zSerie, pSerie, kSerie)
    mag2, ph2, w2, fig, ax = hp.bodeplot(b_serie, a_serie, "FT Serie")
    hp.pzmap1(zSerie, pSerie, "PZ Serie")

    delaySerie = - np.diff(ph2) / np.diff(w2)    # calcul
    hp.grpdel1(w2, delaySerie, 'Delais Serie')

###############################################################################
def probleme2():
    """
    Framework pour le problème 2

    :return:
    """
    
    #Question A
    
    #Frequence de coupure des filtres, a ajuster
    wc = 2 * np.pi * 5000      # fréquence de coupure rad/s
    
    #Generez les filtres butterworth avec la fonction signal.butter, sous quel forme est la sortie (polynome ou pole-zero-gain)?
    #Transformer la sortie de signal.butter dans l'autre representation (vous pouvez vous inspirer de la question A du probleme 1)
    b1, a1 = signal.butter(2, wc, 'lowpass', analog=True)
    b2, a2 = signal.butter(2, wc, 'highpass', analog=True)

    z1, p1, k1 = signal.tf2zpk(b1, a1)
    z2, p2, k2 = signal.tf2zpk(b2, a2)

    #Affichage des polynomes
    print(f'Passe-bas Numérateur {b1}, Dénominateur {a1}')  # affiche les coefficients correspondants au filtre
    print(f'Passe-haut Numérateur {b2}, Dénominateur {a2}')  # affiche les coefficients correspondants au filtre

    #Questions B et C

    #Génère une onde carrée, ajuster la frequence
    fsquare = 1000  # Hz
    t, step = np.linspace(0, .01, 5000, retstep=True)
    u1 = signal.square(2 * np.pi * fsquare * t, 0.5)

    #Gain de chacune des bandes, ajuster au besoin
    lowk = 1 * k1
    highk = 1 * k2

    temp1 = signal.lsim((z1, p1, lowk), U=u1, T=t)
    hp.timeplt1(t, u1, temp1[0], temp1[1], '1')

    temp2 = signal.lsim((z2, p2, highk), U=u1, T=t)
    hp.timeplt1(t, u1, temp2[0], temp2[1], '2')


    #En vous inspirant de la question C du probleme 1, mettre ces deux filtres en parallele, afficher le lieu de bode et le délai de groupe
    zPara, pPara, kPara = hp.paratf(z1, p1, lowk, z2, p2, highk)
    b_para, a_para = signal.zpk2tf(zPara, pPara, kPara)
    mag1, ph1, w1, fig, ax = hp.bodeplot(b_para, a_para, "FT Para")
    hp.pzmap1(zPara, pPara, "PZ Para")

    delayPara = - np.diff(ph1) / np.diff(w1)
    #hp.grpdel1(w1, delayPara, 'Delais Para2')

    #Simuler la sortie de chacun des filtres (passe-haut, passe-bas et les deux en parallele) avec la fonction lsim
    temp3 = signal.lsim((zPara, pPara, kPara), U=u1, T=t)
    hp.timeplt1(t, u1, temp3[0], temp3[1], '3')
    
    #Question D
    
    # génère une entrée sinusoïdale à la même fréquence que l'onde carrée
    fsin = fsquare
    u2 = np.sin(2*np.pi*fsin*t)

    # redéfinit les gains de bande
    lowk = .1 * k1
    highk = 1 * k2

    temp1 = signal.lsim((z1, p1, lowk), U=u2, T=t)
    hp.timeplt1(t, u2, temp1[0], temp1[1], '1')

    temp2 = signal.lsim((z2, p2, highk), U=u2, T=t)
    hp.timeplt1(t, u2, temp2[0], temp2[1], '2')


#Simuler la sortie de chacun des filtres (passe-haut, passe-bas et les deux en parallele) avec la fonction lsim et l'entre u2
    
    #Question E
    #Inspirer vous du probleme 1 pour afficher le lieu de bode et le delais de groupe des filtres conçus
    
    
###############################################################################
def main():
    # décommentez la ou les lignes correspondant aux fonctions que vous voulez exécuter
    # exampleRacines()
    # exampleBode()
    # exampleButterworth()
    # probleme1()
    probleme2()
    plt.show()


#####################################
# Manière standardisée d'exécuter ce que vous désirez lorsque vous "exécutez" le fichier
# permet d'importer les fonctions définies ici comme un module dans un autre fichier .py
# voir par exemple le fichier helper.py qui n'exécutera jamais rien tout seul
# et dont les fonctions sont importées ici à titre de support pour ce qu'on essaie de faire
# pour l'instant ne rien modifier ci-dessous
if __name__ == '__main__':
    main()
