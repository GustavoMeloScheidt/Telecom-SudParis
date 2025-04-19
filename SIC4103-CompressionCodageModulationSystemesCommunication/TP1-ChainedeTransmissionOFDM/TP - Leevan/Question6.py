# -*- coding: utf-8 -*-
# Nom du fichier: Question6.py
# Ce script implémente la chaîne de transmission OFDM
import numpy as np
import matplotlib.pyplot as plt

import ofdm_fonctions as ofdm

################################################################################
# Définition des paramètres de la transmission
################################################################################
# duree utile (s)
Tu=224e-6
# nombre total de porteuses
N=2048
# indice de la premiere sous-porteuse utile
Kmin=0
# indice de la derniere sous-porteuse utile
Kmax=1704
# periode d'échantillonnage (s)
Ts=Tu/N
# duree de l'intervalle de garde (s)
Delta=Tu/8
# nombre d'échantillons de l'intervalle de garde
L=int(Delta/Ts)

# indices des porteuses pilotes = 0,12,24,..,Kmax
PP=np.arange(0,Kmax+1,12)

# Espacement entre sous-porteuses (Hz)
Cs=1/Tu
# Largeur de la bande utile (Hz)
B=(Kmax-Kmin+1)*Cs

# Energie moyenne par symbole QPSK
Es=1
# Energie moyenne par symbole OFDM
E=Es*(Kmax-Kmin+1)
# Rapport signal-sur-bruit (dB)
EsN0dB=20
# Rapport signal-sur-bruit (échelle linéaire)
EsN0=np.power(10,EsN0dB/10)
# Densité spectrale monolatérale du bruit
N0=Es/EsN0

# retard (s) correspondant a un délai de propagation de 10 km
tau=10e3/3e8
# nombre d'échantillons retardés
theta=int(np.floor(tau/Ts))
# retard fractionnaire
e=(tau-theta*Ts)/Ts

# décalage en frequence (Hz)
Df=1000.0
# nombre de symboles OFDM
T=2

# initialiser le signal émis
s = np.array([])
# initialiser les symboles QPSK émis
symb_QPSK = np.array([])
# pour chaque symbole OFDM
for i in range(T):
    # génération des symboles QPSK pour les 1705 sous-porteuses non-éteintes
    QPSK=ofdm.gen_QPSK(Kmax-Kmin+1)
    # concaténation des symboles QPSK
    symb_QPSK=np.append(symb_QPSK,QPSK)
    # modulation OFDM et introduction du retard fractionnaire
    m=ofdm.modulation_OFDM(np.sqrt(Es)*QPSK,N,L,e)
    # concaténation des symboles OFDM
    s=np.append(s,m)

# introduction du retard
s=np.append(np.zeros(theta),s)
# introduction du décalage en fréquence
t=np.arange(len(s))*Ts
s=s*np.exp(1j*2.0*np.pi*Df*t)
# génération de la réponse impulsionnelle du canal
# (Dirac ou Rayleigh avec profil d'intensité exponentiellement décroissant)
# d'étalement temporel Tm échantillons
Tm=10
c=ofdm.reponse_canal(L,1,Tm)

# génération d'un BABG complexe centré de variance N0
n=np.random.normal(0,np.sqrt(N0/2),size=len(s)+len(c)-1)+\
    1j*np.random.normal(0,np.sqrt(N0/2),size=len(s)+len(c)-1)
# signal recu dans le domaine temporel
y=np.convolve(s,c)+n

###########################################################
# Calcul de la réponse fréquentielle du canal
###########################################################
# axe fréquentiel (Hz)
f=np.arange(N)/N*(1/Ts)
# réponse fréquentielle du canal
H= np.fft.fft(c,N)

###########################################################
# Calcul de la métrique temporelle pour d=0,...,N-1
###########################################################
""" code python manquant pour définir le sous-programme metrique_temporelle"""
P=ofdm.metrique_temporelle(y,L,N,N)

# Tracé de la métrique temporelle
###########################################################
# module de la métrique temporelle en fonction du délai
#plt.plot(np.arange(N),np.abs(P),'b')
# phase de la métrique temporelle en fonction du délai
#plt.plot(np.arange(N),-np.angle(P)/(2.0*np.pi*N*Ts),'r')
#plt.xlabel('délai (échantillons)')
#plt.ylabel('métrique temporelle')
#plt.show()
# estimation du retard (en échantillons)
theta_est = np.where(np.abs(P)==(np.amax((np.abs(P)))))[0]
print('theta= ', theta, ' theta_est= ', theta_est)
# estimation du décalage fréquentiel (en Hz)
Df_est=-1*np.angle(P[theta_est])/(2.0*np.pi*N*Ts)
print('Df= ', Df, ' Df_est= ', Df_est)

################################################################################
# synchro temporelle/fréquentielle+égalisation fréquentielle+décision optimale
# pour un canal de Dirac et e quelconque
################################################################################
for i in range(T):
    # synchronisation temporelle + élimination de l'intervalle de garde
    # pour le i-ème symbole OFDM
    k=np.arange(theta_est+L,theta_est+L+N)+i*(N+L)
    # synchronisation fréquentielle du i-ème symbole OFDM
    z=y[k]*np.exp(-1j*2.0*np.pi*Df_est*k*Ts)
    # démodulation du i-ème symbole OFDM
    Y_tilde=np.fft.fft(z,N)/np.sqrt(N)
    # conserver uniquement les sous-porteuses non-éteintes du i-ème symbole OFDM
    Y=Y_tilde[0:Kmax-Kmin+1];
    # symboles QPSK émis par la i-ème symbole OFDM
    QPSK=symb_QPSK[np.arange(Kmax-Kmin+1)+i*(Kmax-Kmin+1)]
    # récupération des symboles pilotes connus du récepteur
    QPSK_pilotes= QPSK[PP]
    # estimation du la réponse fréquentielle du canal
    H_est=ofdm.estimation_canal(Y,QPSK_pilotes,PP,Es,N0)
    # décision optimale des symboles QPSK émis par le i-ème symbole OFDM
    QPSK_est=ofdm.decision(Y/H_est/np.sqrt(Es))
    # comparer symboles démodulés et symboles émis par le i-ème symbole OFDM
    erreur=np.where(np.absolute(QPSK_est-QPSK)>1e-10)[0]
    # fréquences des sous-porteuses erronées sur le i-ème symbole OFDM
    print('fréquences des sous-porteuses erronées (Hz)= ',erreur*Cs)

    # tracé de la réponse fréquentielle estimée du canal
    plt.stem(f,np.abs(H)**2,'b',markerfmt='bo')
    plt.stem(f[0:Kmax-Kmin+1],np.abs(H_est)**2,'r',markerfmt='ro')
    plt.xlabel('fréquence (Hz)')
    plt.ylabel('module au carré de la réponse fréquentielle estimée du canal')
    plt.show()
    plt.stem(f,np.angle(H),'b',markerfmt='bo')
    plt.stem(f[0:Kmax-Kmin+1],np.angle(H_est),'r',markerfmt='ro')
    plt.xlabel('fréquence (Hz)')
    plt.ylabel('phase de la réponse fréquentielle estimée du canal')
    plt.show()
