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

################################################################################
# génération de N échantillons de sortie du canal
# aux instants d'échantillonnage optimaux 
################################################################################
# instants d'échantillonnage
k=np.arange(N)
t=k*T+epsilon*T
# génération des échantillons sorties du canal
Y=np.zeros(len(t),dtype='complex')
for i in range(len(t)):
    Y[i]=synch.RX(t[i],T,data,N0,phi,fD,epsilon)

# tracé des échantillons sorties du canal
plt.plot(np.real(Y), np.imag(Y), '+')
# tracé du cercle trigonométrique
theta = np.linspace(0,np.pi*2,100)
plt.plot(np.sqrt(Es)*np.cos(theta), np.sqrt(Es)*np.sin(theta),'-')
plt.xlabel('I')
plt.ylabel('Q')
plt.show()

################################################################################
# Boucle à verouillage de phase pour les paramètres
#             - estimée de phase initiale, phi_chap_0=0
#             - gain de boucle, gamma=0.01
#             - décalage fréquentiel, fD=0.0/T
################################################################################
phi_chap_0=0.0
gamma=0.01
[phi_chap,a_chap]=synch.PhaseEstimation(Y,gamma,phi_chap_0,Es)

# calcul du taux d'erreur symbole SER
SER=len(np.where(np.abs(data-a_chap)>1.0e-10)[0])
print("taux d'erreur symbole (phi_chap_0=0,gamma=0.01)= ",SER)

# tracé de la phase en fonction des instants d'échantillonnage
plt.plot(t,phi+2*np.pi*fD*t)
# tracé de l'estimé de phase en fonction des instants d'échantillonnage
plt.plot(t,phi_chap, 'r')

plt.xlabel('t')
plt.ylabel('estimée de phase')
plt.show()
