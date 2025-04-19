# -*- coding: utf-8 -*-
# Nom du fichier: ofdm_fonctions.py
import numpy as np

#####################################################################
#
#  Génération de N symboles QPSK aléatoires
#  dans l'alphabet {-1-1j,-1+1j,+1-1j,+1+1j}/sqrt(2)
#
#  entrées:
#  - N: nombre symboles
#
#  sorties:
#  - symb_QPSK[N]: vecteur contenant les symboles
#
####################################################################
def gen_QPSK(N):
   symb_QPSK=(2.0*np.round(np.random.rand(1,N))-1+\
              1j*(2.0*np.round(np.random.rand(1,N))-1))/np.sqrt(2)
   # transformer matrice de dimension (1,N) en un vecteur de dimension (N,)
   symb_QPSK=np.squeeze(symb_QPSK)
   
   return symb_QPSK

#####################################################################
#
#  Modulation OFDM
#
#  entrées:
#  - QPSK: vecteur des symboles QPSK (autant que de sous-porteuses non-eteintes)
#  - N: nombre de porteuses total
#  - L: nombre d'echantillons de l'intervalle de garde
#  - e: retard fractionnaire
#
#  sorties:
#  - symb_QPSK[N]: vecteur contenant les symboles
#
####################################################################
def modulation_OFDM(QPSK,N,L,e):
   # IFFT normalisée
   retard=np.exp(-1j*2.0*np.pi*e*np.arange(len(QPSK))/N)
   s=N*np.fft.ifft(QPSK*retard,N)/np.sqrt(N)
   # insertion de l'intervalle de garde
   tmp=s[len(s)-L:len(s)]
   s=np.append(tmp,s)

   return s

#####################################################################
#
#  Génération de la réponse impulsionnelle du canal
#
#  entrées:
#  - L: longueur de la réponse impulsionnelle (en nombre d'échantillons)
#  - mode: type de canal
#    mode=0 correspond a l'absence de canal 
#    mode=1 correspond a la presence d'un canal avec une reponse impulsionnelle
#           exponentiellement decroissante d'energie unite
#           d'étalement temporel Tm echantillons
#  - Tm: étalement temporel du canal (en nombre d'échantillons)
#
#  sorties:
#  - c[L]: vecteur des coefficients de la réponse impulsionnelle du canal
#
####################################################################
def reponse_canal(L,mode,Tm):
   # initialisation des coefficients de la réponse impulsionnelle du canal
   c=np.zeros(L,dtype='complex')
   # generation d'un Dirac
   if mode==0:
      c[0]=1.0
   # génération d'une réponse exponentiellement decroissante d'énergie unité
   else:
      # énergie de chaque coefficients de la réponse impulsionnelle du canal
      for k in range(L):
         c[k]=np.exp(-k/Tm)
      # normalisation de l'énergie moyenne du canal à 1
      c=c/np.sqrt(np.sum(c))
      # canal de Rayleigh (trajets multiples indépendants)
      tmp=np.random.normal(0,1,size=len(c))+\
         1j*np.random.normal(0,1,size=len(c))
      c=np.sqrt(0.5*c)*tmp

   return c

#####################################################################
#
#  Estimateur de canal dans le domaine frequenciel
#
#  entrées:
#  - Y[N]: observations bruitées sur les sous-porteuses non-éteintes
#  - QPSK_pilotes: symboles pilotes connus du récepteur
#  - PP: indices des porteuses pilotes
#  - Es: Energie moyenne par symbole
#  - N0: variance du bruit d'observation
#
#  sorties:
#  - H_est[N]:réponse fréquentielle du canal sur les sous-porteuses non-éteintes
#
####################################################################
def estimation_canal(Y,QPSK_pilotes,PP,Es,N0):
   # nombre de sous-porteuses non-éteintes
   N=len(Y)

   # création des matrices d'observation
   alphak=np.zeros(N,dtype='complex')
   compt=0
   for k in range(N):
      # dans le cas d'une porteuse pilote
      if k==PP[compt]:
         alphak[k]=QPSK_pilotes[compt]*np.sqrt(Es)
         compt=compt+1
      else:
         alphak[k]=0.0

   # initialisation du filtre de Kalman
   x0=0
   P0=1
   R=N0
   rho=0.9
   G=np.sqrt(1-rho**2)
   F=1
   Q=1
   
   # Filtrage de Kalman
   xf=np.zeros((1,N),dtype='complex')
   Pf=np.zeros((1,1,N),dtype='complex')
   P_pred=np.zeros((1,1,N),dtype='complex')

   for k in range(N):
      if k==0:
         # initialiser la récursion avant du filtre de Kalman
         [xf[:,k],Pf[:,:,k],P_pred[:,:,k]]=\
            Kalman_filter(x0,P0,Y[k],F,G,alphak[k],Q,R)
      else:
         # récursion avant du filtre de Kalman
         [xf[:,k],Pf[:,:,k],P_pred[:,:,k]]=\
            Kalman_filter(xf[:,k-1],Pf[:,:,k-1],Y[k],F,G,alphak[k],Q,R)

   # Lissage de Kalman pour l'interpolation
   xb=np.zeros((1,N),dtype='complex')
   Pb=np.zeros((1,1,N),dtype='complex')

   for k in range(N-1,-1,-1):
      if k==N-1:
         # initialiser la récursion arrière du lisseur de Kalman
         xb[:,N-1]=xf[:,N-1]
         Pb[:,:,N-1]=Pf[:,:,N-1]
      else:
         # récursion arrière du lisseur de Kalman
         [xb[:,k],Pb[:,:,k]]=Kalman_smoother(xf[:,k],Pf[:,:,k],P_pred[:,:,k+1],F,xb[:,k+1],Pb[:,:,k+1])

   
   # résultat
   H_est=xb[0,:]
   
   return H_est

#####################################################################
#
#  Filtre de Kalman
#
#  entrées:
#  - xk_1k_1: estimée à l'instant k-1
#  - Pk_1k_1: covariance de l'erreur d'estimation à l'instant k-1
#  - yk: observation à l'instant k
#  - Fk: matrice de transition de l'état à l'instant k
#  - Gk: matrice de transition du bruit de processus à l'instant k
#  - Hk: matrice d'observation à l'instant k
#  - Q: matrice de covariance du bruit de processus à l'instant k
#  - R: matrice de covariance du bruit d'observation à l'instant k
#
#  sorties:
#  - xk: estimée à l'instant k
#  - Pkk: covariance de l'erreur d'estimation à l'instant k
#  - Pkk_1: covariance de l'erreur de prédiction à l'instant k
#
####################################################################
def Kalman_filter(xk_1k_1,Pk_1k_1,yk,Fk,Gk,Hk,Q,R):
   # predicted estimate
   xkk_1=Fk*xk_1k_1

   # covariance de l'erreur de prédiction à l'instant k
   Pkk_1=Fk*Pk_1k_1*np.conj(np.transpose(Fk))+Gk*Q*np.conj(np.transpose(Gk))

   # gain de Kalman
   if np.isscalar(R)==True:
      Kk=Pkk_1*np.conj(np.transpose(Hk))\
         /(Hk*Pkk_1*np.conj(np.transpose(Hk))+R)
   else:
      Kk=Pkk_1*np.conj(np.transpose(Hk))*\
         np.linalg.inv(Hk*Pkk_1*np.conj(np.transpose(Hk))+R)

   # estimée à l'instant k
   xkk=xkk_1+Kk*(yk-Hk*xkk_1)

   # covariance de l'erreur d'estimation à l'instant k
   Pkk=Pkk_1-Kk*Hk*Pkk_1

   return xkk,Pkk,Pkk_1

#####################################################################
#
#  Lisseur de Kalman (recursion arrière)
#
#  entrées:
#  - xk: estimée à l'instant k
#  - Pkk: covariance de l'erreur d'estimation à l'instant k
#  - Pkk1: covariance de l'erreur de prédiction à l'instant k+1
#  - Fk: matrice de transition de l'état à l'instant k
#  - xk1l: estimée lissée à l'instant k+1
#  - Pk1l: covariance de l'erreur d'estimation à l'instant k+1
#
#  sorties:
#  - xkl: estimée lissée à l'instant k
#  - Pkl: covariance de l'erreur d'estimation à l'instant k
#
####################################################################
def Kalman_smoother(xkk,Pkk,Pk1k,Fk,xk1l,Pk1l):
   # matrice Sk
   if np.isscalar(Pk1k)==True:
      Sk=Pkk*np.conj(np.transpose(Fk))/Pk1k
   else:
      Sk=Pkk*np.conj(np.transpose(Fk))*np.linalg.inv(Pk1k)

   # estimée lissée
   xkl=xkk+Sk*(xk1l-Fk*xkk)

   # covariance de l'erreur de lissage à l'instant k
   Pkl=Pkk+Sk*(Pk1l-Pk1k)*np.conj(np.transpose(Sk))

   return xkl,Pkl

#####################################################################
#
#  Calcul de la métrique temporelle pour d=0,1,..nb-1
#
#  entrées:
#  - y: sortie du canal dans le domaine temporel
#  - L: nombre d'échantillons de l'intervalle de garde
#  - N: nombre total de sous-porteuses
#  - nb: nombre de valeurs pour le délai
#
#  sorties:
#  - P[nb]: vecteur des valeurs de la métrique temporelle 
#  - y[0:T*(N+L)+len(c)-2] : signal reçu en sortie du canal
#  -
####################################################################
def metrique_temporelle(y,L,N,nb):
   # initialiser un vecteur pour les valeurs de la métrique temporelle 
   P=np.zeros(nb,dtype='complex')

   # pour chaque valeur du délai
   for d in range(nb):
      P[d]=0.0
      for m in range(L):
         P[d]+=y[m+d]*np.conj(y[m+d+N])
         
   return P

#####################################################################
#
#  Décision optimale au sens du maximum de vraisemblance
#  pour la constellation MDP-4
#
#  entrées:
#  - r: échantillons bruité
#
#  sorties:
#  - res: décisions optimale
#
####################################################################
def decision(r):
   d =(np.sign(np.real(r)) + 1j*np.sign(np.imag(r)))/np.sqrt(2)
   return (d)
