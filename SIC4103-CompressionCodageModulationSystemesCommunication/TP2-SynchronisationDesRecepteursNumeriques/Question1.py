# -*- coding: utf-8 -*-
# Nom du fichier: TP.py
# Ce script implémente la boucle à verouillage de phase
import numpy as np
import matplotlib.pyplot as plt

import synch_fonctions as synch

################################################################################
# Définition des paramètres de la transmission
################################################################################
# Nombre de symboles
N=1000
# Durée symbole
T=1
# Rapport signal-sur-bruit (dB)
EsN0dB=20
# Rapport signal-sur-bruit (échelle linéaire)
EsN0=np.power(10,EsN0dB/10)
# Energie moyenne par symbole
Es=1
# Densité spectrale monolatérale du bruit
N0=Es/EsN0
# Déphasage introduit par le canal
phi=np.pi/8
# Décalage fréquentiel introduit par le canal
fD=0.0
# Retard fractionnaire introduit par le canal
epsilon=-0.4

################################################################################
# génération de N symboles MDP-4 aléatoires 
################################################################################
data=synch.gendata(Es,N)
# tracé de la constellation des symboles dans le plan complexe
#""" code python manquant """

plt.plot(np.real(data), np.imag(data), '+')

# tracé du cercle trigonométrique
theta = np.linspace(0,np.pi*2,100)
plt.plot(np.sqrt(Es)*np.cos(theta), np.sqrt(Es)*np.sin(theta),'-')
plt.xlabel('I')
plt.ylabel('Q')
plt.show()
