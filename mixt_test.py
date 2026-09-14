import os
import math
import numpy as np
import scipy as sp
import random
from sympy import symbols, prod
import matplotlib.pyplot as plt
import time
from itertools import combinations, product
from collections import deque

np.set_printoptions(precision=5, linewidth=150, suppress=True)


# Etats de la base de Fock

class Ket:
    "Cree un objet ket representant un etat de la base de Fock"

    def __init__(self, vector, typ, name=None, compteur=0):
        self.type = typ    # "r" ou "k"
        self.vector = np.array(vector)    # liste d'occupation dans l'ordre normal up-down
        self.name = name    # decimal (representant vector) + typ (ex: 165r)
        self.compteur = compteur    # indice dans la matrice (etats mixtes)

    def number(self):
        "Retourne l'indice matriciel d'un etat selon son vector et son typ"

        return int("".join(map(str, self.vector)), 2) + int(self.type == "k") * 4**(len(self.vector)//2)

    def __repr__(self):
        "Retourne le name de l'etat"

        return self.name if self.name is not None else "Ket"


# Fonctions pour determiner les blocs hamiltoniens

def connected_component(H, start):

    n = H.shape[0]
    visited = set([start])
    queue = deque([start])

    while queue:
        
        i = queue.popleft()
        neighbors = np.where(H[i] != 0)[0]

        for j in neighbors:
            if j not in visited:
                
                visited.add(int(j))
                queue.append(j)

    return visited

def all_components(H):
    
    n = H.shape[0]
    visited = set()
    components = []

    for i in range(n):
        if i not in visited:
            
            comp = connected_component(H, i)
            components.append(comp)
            visited |= comp

    return components




# Creation des matrices completes H et S 

def m_get_H_S(N, t=1, U=2):
    """
    Retourne les matrices H et S  completes en base d'etats mixtes
    d'un systeme a N dimensions pour des parametres t et U
    """

    H_file = f"m_H_{N}_{t}_{U}.npy"
    S_file = f"m_S_{N}.npy"

    # Si les matrices sont construites 
    if os.path.exists(H_file) and os.path.exists(S_file):

        H = np.load(H_file)
        S = np.load(S_file)
        
        return H, S

    # Si les matrices ne sont pas construites
    etats = np.arange(4**(2*N))

    states = []

    compteur = 0

    # Creation de tous les etats mixtes
    for element in etats:

        lst = list(map(int, format(element, f'0{4*N}b')))

        if sum(lst[:2*N]) != N/2 or sum(lst[2*N:]) != N/2:
            continue

        ket = Ket(lst, None, str(element), compteur=compteur)
        states.append(ket)

        compteur += 1

    # Creation des matrices completes
    _, H, S, _, _ = m_ground_energy(states, N, t, U)

    np.save(H_file, H)
    np.save(S_file, S)

    return H, S

def get_H_S(N, t=1, U=2):
    """
    Retourne les matrices H et S completes dans une base mixte
    d'un systeme a N dimensions pour des parametres t et U
    """

    H_file = f"H_{N}_{t}_{U}.npy"
    S_file = f"S_{N}.npy"

    # Si les matrices sont construites
    if os.path.exists(H_file) and os.path.exists(S_file):

        H = np.load(H_file)
        S = np.load(S_file)
        
        return H, S

    # Si les matrices ne sont pas construites
    H, S = complete_matrixes(N, t, U)

    np.save(H_file, H)
    np.save(S_file, S)

    return H, S



# Diagonalisation d'une base surcomplete

def gen_diagonalization(H, S, eps=1e-6):
    """
    Diagonalise la matrice hamiltonienne (H) dans une 
    base mixte surcomplete de metrique S et retourne 
    les valeurs propres (eigenvalues) et les vecteurs 
    propres exprimes dans la base mixte surcomplete de 
    depart (eigenvectors)
    """

    # Directions lineairement dependantes
    eig_S, U = np.linalg.eigh(S)
    mask = eig_S > eps

    # Retrait des directions lineairement dependantes
    eig_S = eig_S[mask]
    U = U[:, mask]

    # Projection dans la base complete W
    W = U @ np.diag(1/np.sqrt(eig_S))
    H_red = W.conj().T @ H @ W

    # Diagonalisation dans la base complete
    eigenvalues, Y = np.linalg.eigh(H_red)

    # Projection des vecteurs propres dans la base surcomplete
    eigenvectors = W @ Y

    return eigenvalues, eigenvectors



# Modele de Hubbard dans une base d'etats mixtes

def split_state(state, N):
    """
    Prend en entree un etat mixte et le nombre
    de sites du systeme (N) et le separe en ses 
    parties r/k et up/down
    """

    r_up, k_up, r_down, k_down = (state[i*N:(i+1)*N] for i in range(4))

    return r_up, k_up, r_down, k_down

def replace_zero_combinations(k, r, N):
    """
    Prend en entree la partie en k et la partie en 
    r d'un spin donne d'un etat mixte dans un systeme a
    N sites. Transforme la partie r en k (Transformee 
    de Fourier) et retourne la combinaison lineaire 
    des etats globaux (pour le spin donne) possibles en 
    k, soient les vecteurs (k_states) et les 
    coefficients de Fourier (c)
    """

    k = np.array(k)
    n = sum(r)
    
    zero_pos = np.where(k == 0)[0]
    one_pos  = np.where(k == 1)[0]

    k_states = []
    sigma   = []
    combs = []

    # Combinaisons possibles de k
    for c in combinations(zero_pos, n):
        
        # Signe de la permutation
        compteur = np.sum(one_pos < np.array(c)[:, None])
        sigma.append(compteur)

        # Creation de l'etat k global
        new_k = k.copy()
        new_k[list(c)] = 1
        k_states.append(new_k)
    
        # Transformation de r vers k
        combs.append(c)

    r_pos = np.flatnonzero(r)
    factor = 1j * (2*np.pi/N)

    c = []

    # Calcul du coefficient de Fourier (signe inclus)
    for element, s in zip(combs, sigma):

        k_pos = element
        mat = np.exp(factor * np.outer(r_pos, k_pos))
        c.append((-1)**s * np.linalg.det(mat))

    return k_states, c

def met_m(state1, state2, N):
    """
    Prend en entree deux etats mixtes (state1 et
    state2) et la taille du systeme (N) et retourne 
    le produit scalaire entre les deux etats (soit 
    l'element de la matrice S associe a (state1, state2)), 
    la decomposition de chacun des etats mixtes en etats 
    k (k_states1, c1, k_states2, c2) ainsi que le facteur 
    de normalisation (norm) et un indicateur d'erreur en 
    cas d'utilisation de parametres non physiques (err)
    """

    r_up1, k_up1, r_down1, k_down1 = split_state(state1, N)
    r_up2, k_up2, r_down2, k_down2 = split_state(state2, N)

    n_up1 = sum(r_up1) + sum(k_up1)
    n_down1 = sum(r_down1) + sum(k_down1)
    n_up2 = sum(r_up2) + sum(k_up2)
    n_down2 = sum(r_down2) + sum(k_down2)
    
    # Etats mixtes non physiques
    if (n_up1 > N or n_up2 > N or n_down1 > N or n_down2 > N
        or n_up1 != n_up2 or n_down1 != n_down2):

        err = True

        return 0, [], [], [], [], 0, err

    # Decomposition des parties up/down des etats mixtes en k
    k_up_states1, c_up1 = replace_zero_combinations(k_up1, r_up1, N)
    k_down_states1, c_down1 = replace_zero_combinations(k_down1, r_down1, N)
    k_up_states2, c_up2 = replace_zero_combinations(k_up2, r_up2, N)
    k_down_states2, c_down2 = replace_zero_combinations(k_down2, r_down2, N)

    # Expression des etats mixtes en k
    k_states1 = [list(a) + list(b) for a, b in product(k_up_states1, k_down_states1)]
    k_states2 = [list(a) + list(b) for a, b in product(k_up_states2, k_down_states2)]

    # Coefficients de Fourier associes
    c1 = [a*b for a, b in product(c_up1, c_down1)]
    c2 = [a*b for a, b in product(c_up2, c_down2)]

    # Correspondances entre les termes k des deux etats mixtes
    d2 = {}
    for j, b in enumerate(k_states2):
        key = tuple(b)
        d2.setdefault(key, []).append(j)
    matches = [(i, j) for i, a in enumerate(k_states1) for j in d2.get(tuple(a), [])]

    # Calcul du produit scalaire
    somme = sum(np.conj(c1[i]) * c2[j] for i, j in matches)
    norm = N**(-0.5 * (sum(r_up1) + sum(r_down1) + sum(r_up2) + sum(r_down2)))
    err = False

    return norm * somme, k_states1, c1, k_states2, c2, norm, err

def m_ground_energy(states, N, t=1, U=2, mu=0):
    """
    Prend en entree un ensemble d'etats mixtes (states)
    et la taille du systeme (N) avec parametres t, U et mu
    et retourne les valeurs propres du systeme (eigenvalues)
    les vecteurs propres en ligne (eigenvectors.T), les
    matrices H et S (new_H et new_S) ainsi qu'un indicateur
    de surcompletude de la base d'etats mixtes (overfilled)

    *Creation des matrices element par element => diagonalisation*
    """

    # Creation des matrices vides
    dim = len(states)
    H = np.zeros((dim, dim), dtype=complex)
    S = np.zeros((dim, dim), dtype=complex)

    # Creation des matrices (triangulaires superieures) element par element
    for i in range(dim):
        for j in range(i, dim):
             
            # Element de la matrice S + decomposition en k des etats
            S[i, j], k_states1, c1, k_states2, c2, norm, err = met_m(states[i].vector, states[j].vector, N)

            # Cas non physique
            if err == True:               
                H[i, j] = 0
                continue

            # Application de H sur chacune des combinaisons d'etats k
            H[i, j] = 0
            for l in range(len(k_states1)):
                for m in range(len(k_states2)):
                    H[i, j] += np.conj(c1[l]) * c2[m] * ham_k(k_states1[l], k_states2[m], N, t, U, mu)
            H[i, j] = norm * H[i, j]
 
    # Utilisation de la propriete d'hermiticite des matrices   
    new_H = H + H.conj().T
    np.fill_diagonal(new_H, np.diag(H).real)
    new_S = S + S.conj().T
    np.fill_diagonal(new_S, np.diag(S).real)

    # Verification de la surcompletude de la base (valeurs propres de S)
    eig_S = np.linalg.eigh(new_S)[0]

    # Au moins une valeur propre de S nulle => diagonalisation generalisee
    if np.any(np.isclose(eig_S, 0, atol=1e-6)):    
        eigenvalues, eigenvectors = gen_diagonalization(new_H, new_S)        
        overfilled = True

    # Aucune valeur propre de S nulle => diagonalisation par eigh
    else:
        eigenvalues, eigenvectors = sp.linalg.eigh(new_H, new_S)
        overfilled = False

    return eigenvalues, new_H, new_S, eigenvectors.T, overfilled

def m_optimized_ground_energy(states, H, S, N, t=1, U=2, mu=0):
    """
    Prend en entree un ensemble d'etats mixtes (states), 
    les matrices completes du systeme (H et S) et la taille 
    du systeme (N) avec parametres t, U et mu et retourne 
    les valeurs propres du systeme (eigenvalues), les vecteurs 
    propres en ligne (eigenvectors.T), les matrices reduites 
    H et S (new_H et new_S) ainsi qu'un indicateur de 
    surcompletude de la base d'etats mixtes (overfilled)

    *Reduction des matrices completes a la base utilisee => diagonalisation*
    """

    # Liste des indices de la base utilisee
    n = []    
    for element in states:
        n.append(int(element.compteur))

    # Reduction des matrices a la base utilisee
    new_H = H[np.ix_(n, n)]
    new_S = S[np.ix_(n, n)]

    # Verification de la surcompletude de la base (valeurs propres de S)
    eig_S = np.linalg.eigh(new_S)[0]

    # Au moins une valeur propre de S nulle => diagonalisation generalisee
    if np.any(np.isclose(eig_S, 0, atol=1e-6)):       
        eigenvalues, eigenvectors = gen_diagonalization(new_H, new_S)        
        overfilled = True

    # Aucune valeur propre de S nulle => diagonalisation par eigh
    else:
        eigenvalues, eigenvectors = sp.linalg.eigh(new_H, new_S)
        overfilled = False

    return eigenvalues, new_H, new_S, eigenvectors.T, overfilled



# Modele de Hubbard dans une base mixte par la creation de matrices completes

def metrique_r(N):
    """
    Retourne la matrice identite de la taille de
    l'espace de Fock d'un systeme a N sites
    """
        
    return np.identity(4**N, dtype=int)

def metrique_k(N):
    """
    Retourne la matrice identite de la taille de
    l'espace de Fock d'un systeme a N sites
    """

    return np.identity(4**N, dtype=int)

def metrique_rk(N):
    """
    Retourne la metrique rk (matrice des produits 
    scalaires rk) de la taille de l'espace de Fock
    pour un systeme a N sites    
    """
    
    # Creation de la metrique vide
    dim = 4**N
    matrice = np.zeros((dim, dim), dtype=complex)
    
    # Creation des etats (vecteurs) de la base de Fock
    occ = ((np.arange(dim)[:, None] >> np.arange(2*N)) & 1)[:, ::-1]

    # Creation d'une matrice complete de Fourier
    indices = np.arange(2*N)
    F = np.exp(2j * np.pi * np.outer(indices, indices) / N)

    # Creation de la matrice element par element
    for r in range(dim):

        # Occupations en r
        occ_r = occ[r]
        n1_r = np.where(occ_r[:N] == 1)[0]
        n2_r = np.where(occ_r[N:] == 1)[0] + N

        for k in range(dim):

            # Occupations en k
            occ_k = occ[k]
            n1_k = np.where(occ_k[:N] == 1)[0]
            n2_k = np.where(occ_k[N:] == 1)[0] + N

            # Cas d'overlap nul
            if len(n1_r) != len(n1_k) or len(n2_r) != len(n2_k):
                continue

            # Reduction de la matrice de Fourier aux occupations rk
            mat1 = F[np.ix_(n1_r, n1_k)]
            mat2 = F[np.ix_(n2_r, n2_k)]

            # Calcul du produit scalaire (et arrondissement)
            power = 0.5 * (len(n1_r) + len(n2_r))
            el = (1 / N**power) * np.linalg.det(mat1) * np.linalg.det(mat2)
            matrice[r, k] = np.round(el.real, 5) + 1j*np.round(el.imag, 5)

    return matrice

def hamiltonian_r(N, t=1, U=2):
    """
    Retourne la matrice hamiltonienne dans la base
    r de la taille de l'espace de Fock pour un 
    systeme a N sites avec des parametres t et U  
    """

    # Creation de la matrice vide
    dim = 4 ** N
    H = np.zeros((dim, dim), dtype=float)

    # Creation des etats (vecteurs) de la base de Fock
    etats = [list(map(int, format(i, f'0{2*N}b'))) for i in range(dim)]
    index = {tuple(e): k for k, e in enumerate(etats)}    # indice des etats
   
    # Creation des elements de la matrice non nuls
    for element in etats:

        i0 = index[tuple(element)]

        element_up = element[:N]
        element_down = element[N:]
        n_up = sum (element_up)
        n_down = sum(element_down)

        n_U = 0

        for i in range(N):

            # Nombre de doubles occupations
            if element[i] == 1 and element[i + N] == 1:
                n_U += 1
            
            # Saut vers la gauche d'un electron up
  
            if element_up[i] == 1 and element_up[i - 1] == 0:

                # Creation de l'etat apres saut
                new_state = element.copy()
                new_state[i] = 0
                if i != 0:    # saut conventionnel
                    new_state[i - 1] = 1
                    j0 = index[tuple(new_state)]  
                    H[i0, j0] = -t
                elif i == 0:    # saut par symetrie de translation
                    new_state[N - 1] = 1
                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t * (-1) ** (n_up - 1)    # signe de permutation

            # Saut vers la gauche d'un electron down

            if element_down[i] == 1 and element_down[i - 1] == 0:

                # Creation de l'etat apres saut
                new_state = element.copy()
                new_state[i + N] = 0              
                if i != 0:    # saut conventionnel
                    new_state[i + N - 1] = 1
                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t
                elif i == 0:    # saut par symetrie de translation
                    new_state[-1] = 1
                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t * (-1) ** (n_down - 1)    # signe de permutation

            # Saut vers la droite d'un electron up
    
            if element_up[i] == 0 and element_up[i - 1] == 1:

                # Creation de l'etat apres saut
                new_state = element.copy()
                new_state[i] = 1
                if i != 0:    # saut conventionnel
                    new_state[i - 1] = 0
                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t
                elif i == 0:    # saut par symetrie de translation
                    new_state[N - 1] = 0
                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t * (-1) ** (n_up - 1)    # signe de permutation
                    
            # Saut vers la droite d'un electron down

            if element_down[i] == 0 and element_down[i - 1] == 1:

                # Creation de l'etat apres saut
                new_state = element.copy()
                new_state[i + N] = 1                
                if i != 0:    # saut conventionnel
                    new_state[i + N - 1] = 0
                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t
                elif i == 0:    # saut par symetrie de translation
                    new_state[-1] = 0
                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t * (-1) ** (n_down - 1)    # signe de permutation
 
        # Element diagonal
        H[i0, i0] = n_U * U        
   
    return H

def hamiltonian_k(N, t=1, U=2):
    """
    Retourne la matrice hamiltonienne dans la base
    k de la taille de l'espace de Fock a N dimensions
    pour des parametres t et U  
    """
   
    # Creation de la matrice vide
    dim = 4 ** N
    H = np.zeros((dim, dim), dtype=float)

    # Creation des etats (vecteurs) de la base de Fock
    etats = [list(map(int, format(i, f'0{2*N}b'))) for i in range(dim)]
    index = {tuple(e): k for k, e in enumerate(etats)}    # indice des etats
    
    # Creation des elements de la matrice non nuls
    for element in etats:

        i0 = index[tuple(element)]

        element_up = element[:N]
        element_down = element[N:]
        n_up = sum(element_up)
        n_down = sum(element_down)

        somme = 0
        
        # Element diagonal
        for i in range(N):
            somme += np.cos((2*np.pi*i) / N) * (element_up[i] + element_down[i])
        H[i0, i0] = -2*t * somme + U/N * n_up * n_down

        # Termes de potentiel
        for i in range(N):
            if element_up[i] == 1:               
                for j in range(N):
                    if element_up[j] == 0:

                        # Longueur du saut en up (de i vers j)
                        if j > i:
                            q = i - j + N
                            zwizz = 1    # indicateur de signe
                        elif j < i:
                            q = i - j
                            zwizz = 0    # indicateur de signe
                            
                        for k in range(N):
                            # Meme longueur du saut en down (de k vers k-q)
                            if element_down[k] == 0 and element_down[k - q] == 1:

                                    # Creation de l'etat apres saut
                                    new_state = element.copy()
                                    new_state[i] = 0
                                    new_state[j] = 1
                                    new_state[k + N] = 1
                                    if k - q >= 0:    # saut conventionnel
                                        new_state[k - q + N] = 0
                                        j0 = index[tuple(new_state)]
                                        H[i0, j0] = U/N * (-1) ** (sum(element_up[:i]) + sum(element_up[:j]) + sum(element_down[:k]) + sum(element_down[:(k-q)]) - zwizz - 1)     
                                    elif k - q < 0:    # saut par symetrie de translation
                                        new_state[k - q] = 0
                                        j0 = index[tuple(new_state)]
                                        H[i0, j0] = U/N * (-1) ** (sum(element_up[:i]) + sum(element_up[:j]) + sum(element_down[:k]) + sum(element_down[:(N+k-q)]) - zwizz)
                                        
    return H

def hamiltonian_rk(N, t=1, U=2):
    """
    Retourne la matrice hamiltonienne (a un facteur
    metrique pres) dans la base mixte rk de la taille 
    de l'espace de Fock pour un systeme a N sites 
    avec des parametres t et U 
    """

    # Creation des matrices vides
    dim = 4 ** N
    H_t = np.zeros(dim, dtype=float)
    H_U = np.zeros(dim, dtype=float)

    # Creation des etats (vecteurs) de la base de Fock
    etats = [list(map(int, format(i, f'0{2*N}b'))) for i in range(dim)]
    index = {tuple(e): k for k, e in enumerate(etats)}

    # Creation des elements non nuls des matrices
    for i in range(dim):     
        for j in range(N):    
            H_t[i] += -2*t * np.cos((2*np.pi*j)/N) * (etats[i][j] + etats[i][j+N])    # diagonal en k
            H_U[i] += U * etats[i][j] * etats[i][j+N]    # diagonal en r

    return np.add.outer(H_t, H_U)

def complete_matrixes(N, t=1, U=2):
    """
    Retourne les matrices H et S completes
    dans l'espace de Fock pour un systeme de
    N sites avec des parametres t et U
    """

    # Creation de la matrice complete S
    S_r = metrique_r(N)
    S_k = metrique_k(N)
    S_rk = metrique_rk(N)
    S_kr = S_rk.conj().T    # par la propriete d'hermiticite
    S = np.block([[S_r,  S_kr], [S_rk, S_k]])

    # Creation de la matrice complete H
    H_r = hamiltonian_r(N, t, U)
    H_k = hamiltonian_k(N, t, U)
    H_rk = np.multiply(hamiltonian_rk(N, t, U), S_rk)
    H_kr = H_rk.conj().T    # par la propriete d'hermiticite
    H = np.block([[H_r,  H_kr], [H_rk, H_k]])

    return H, S

def optimized_ground_energy(states, H, S, N, t=1, U=2):
    """
    Prend en entree un ensemble d'etats de la base mixte
    (states), les matrices completes du systeme (H et S) 
    et la taille du systeme (N) avec parametres t et U et 
    retourne les valeurs propres du systeme (eigenvalues), 
    les vecteurs propres en ligne (eigenvectors.T), les 
    matrices reduites H et S (new_H et new_S) ainsi qu'un 
    indicateur de surcompletude de la base mixte (overfilled)

    *Reduction des matrices completes a la base utilisee => diagonalisation*
    """

    # Liste des indices de la base utilisee
    n = []
    for element in states:
        if element.type == "r":
            n.append(element.number())
        if element.type == "k":
            n.append(element.number())

    # Reduction des matrices a la base utilisee
    new_H = H[np.ix_(n, n)]
    new_S = S[np.ix_(n, n)]

    # Verification de la surcompletude de la base (valeurs propres de S)
    eig_S = np.linalg.eigh(new_S)[0]

    # Au moins une valeur propre de S nulle => diagonalisation generalisee
    if np.any(np.isclose(eig_S, 0, atol=1e-4)):       
        eigenvalues, eigenvectors = gen_diagonalization(new_H, new_S)        
        overfilled = True

    # Aucune valeur propre de S nulle => diagonalisation par eigh
    else:
        eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(new_H, 1, new_S, which="SA")
        overfilled = False

    return eigenvalues, new_H, new_S, eigenvectors.T, overfilled



# Modele de Hubbard dans une base mixte par la creation de matrices element par element

def met_r(state1, state2, N):
    """
    Retourne l'element de la matrice S dans la
    base r (state1, state2) pour un systeme a N sites
    """

    return int(np.array_equal(state1, state2))

def met_k(state1, state2, N):
    """
    Retourne l'element de la matrice S dans la
    base k (state1, state2) pour un systeme a N sites
    """

    return int(np.array_equal(state1, state2))

def met_rk(order, state1, state2, N):
    """
    Retourne l'element de la matrice S dans la
    base mixte "rk" ou "kr" (selon order) 
    (state1, state2) pour un systeme a N sites
    """
    
    state1_up = state1[:N]
    state1_down = state1[N:]
    state2_up = state2[:N]
    state2_down = state2[N:]
    n_up = np.count_nonzero(state1_up)
    n_down = np.count_nonzero(state1_down)

    # Cas d'overlap nul
    if (n_up != np.count_nonzero(state2_up) or
        n_down != np.count_nonzero(state2_down)):
        return 0

    # Produit scalaire <r|k>
    if order == "rk":
        r_up = np.flatnonzero(state1_up)
        k_up = np.flatnonzero(state2_up)
        r_down = np.flatnonzero(state1_down)
        k_down = np.flatnonzero(state2_down)
        factor = -1j * (2*np.pi/N)    # transformation de k vers r

    # Produit scalaire <k|r>
    elif order == "kr":
        k_up = np.flatnonzero(state1_up)
        r_up = np.flatnonzero(state2_up)
        k_down = np.flatnonzero(state1_down)
        r_down = np.flatnonzero(state2_down)
        factor = 1j * (2*np.pi/N)    # transformation de k vers r

    # Produit scalaire
    det_up = np.linalg.det(np.exp(factor * np.outer(r_up, k_up)))
    det_down = np.linalg.det(np.exp(factor * np.outer(r_down, k_down)))
    return N**(-(n_up+n_down)/2) * det_up * det_down

def ham_r(state1, state2, N, t=1, U=2, mu=0):
    """
    Retourne l'element de la matrice H dans la
    base r (state1, state2) pour un systeme a N sites
    avec des parametres t, U et mu
    """

    # Cas diagonal
    if np.array_equal(state1, state2):
        state_up = state1[:N]
        state_down = state1[N:]
        n_U = 0

        # Nombre de doubles occupations
        for i in range(N):                      
            if state_up[i] == 1 and state_down[i] == 1:
                n_U += 1

        H = U * n_U - mu * (sum(state1))
        return H

    # Cas hors diagonale    
    state1_up = state1[:N]
    state1_down = state1[N:]
    n_up1 = sum(state1_up)
    n_down1 = sum(state1_down)
    state2_up = state2[:N]
    state2_down = state2[N:]
    n_up2 = sum(state2_up)
    n_down2 = sum(state2_down)

    # Cas non physique (non conservation de N_e/S_z)
    if n_up1 != n_up2 or n_down1 != n_down2:
        H = 0
        return H

    # Differences entre state1 et state2
    index_up = []
    index_down = []
    for i in range(N):
        if state1_up[i] != state2_up[i]:
            index_up.append(i)
        if state1_down[i] != state2_down[i]:
            index_down.append(i)
    
    # Cas de plusieurs sauts
    if not ((len(index_up) == 2 and len(index_down) == 0) or (len(index_up) == 0 and len(index_down) == 2)):
        H = 0
        return H

    # Saut d'electron up
    if len(index_up) == 2:
        if index_up[1] - index_up[0] == 1:    # saut conventionnel
            zwizz = 0
        elif index_up[1] - index_up[0] == N - 1:    # saut par symetrie de translation
             zwizz = n_up1 - 1
        else:    # saut de plus d'un site
            H = 0
            return H

    # Saut d'electron down
    elif len(index_down) == 2:
        if index_down[1] - index_down[0] == 1:    # saut conventionnel
            zwizz = 0
        elif index_down[1] - index_down[0] == N - 1:    # saut par symetrie de translation
             zwizz = n_down1 - 1
        else:    # saut de plus d'un site
            H = 0
            return H

    H = -t * (-1) ** (zwizz)
    return H

def ham_k(state1, state2, N, t=1, U=2, mu=0):
    """
    Retourne l'element de la matrice H dans la
    base k (state1, state2) pour un systeme a N sites
    avec des parametres t, U et mu
    """

    # Cas diagonal
    if np.array_equal(state1, state2):
        state_up = state1[:N]
        state_down = state1[N:] 
        n_up = sum(state_up)
        n_down = sum(state_down)
        somme = 0
        
        # Partie cinetique
        for i in range(N):
            somme += np.cos((2*np.pi*i) / N) * (state_up[i] + state_down[i])
        
        H = -2*t * somme + U/N * n_up * n_down - mu * sum(state1)
        return H

    # Cas hors diagonale
    state1_up = state1[:N]
    state1_down = state1[N:]
    n_up1 = sum(state1_up)
    n_down1 = sum(state1_down)
    state2_up = state2[:N]
    state2_down = state2[N:]
    n_up2 = sum(state2_up)
    n_down2 = sum(state2_down)

    # Cas non physique (non conservation de N_e/S_z)
    if n_up1 != n_up2 or n_down1 != n_down2:
        H = 0
        return H
        
    # Differences entre state1 et state2
    k_up = []
    k_up_dag = []
    k_down = []
    k_down_dag = []
    for i in range(N):
        if state1_up[i] == 0 and state2_up[i] == 1:
            k_up.append(i)
        elif state1_up[i] == 1 and state2_up[i] == 0:
            k_up_dag.append(i)
        if state1_down[i] == 0 and state2_down[i] == 1:
            k_down.append(i)
        elif state1_down[i] == 1 and state2_down[i] == 0:
            k_down_dag.append(i)

    # Cas de sauts non physiques
    if len(k_up_dag) != 1 or len(k_down_dag) != 1 or len(k_up) != 1 or len(k_down) != 1:
        H = 0
        return H

    # Longueur du saut d'electron up (de k_up_dag vers k_up)
    if k_up_dag[0] > k_up[0]:
        q_up = k_up_dag[0] - k_up[0]
        zwizz_up = 0    # indicateur de signe
    elif k_up_dag[0] < k_up[0]:
        q_up = k_up_dag[0] - k_up[0] + N 
        zwizz_up = 1    # indicateur de signe

    # Longueur du saut d'electron down (de k_down_dag vers k_down)
    if k_down_dag[0] > k_down[0]:
        q_down = k_down[0] - k_down_dag[0] + N
        zwizz_down = 0    # indicateur de signe
    elif k_down_dag[0] < k_down[0]:
        q_down = k_down[0] - k_down_dag[0]
        zwizz_down = 1    # indicateur de signe

    # Cas de longueurs de saut differentes en up/down
    if q_up != q_down:
        H = 0
        return H

    H = U/N * (-1) ** (sum(state1_up[:k_up[0]]) + sum(state1_up[:k_up_dag[0]]) + sum(state1_down[:k_down[0]]) + sum(state1_down[:k_down_dag[0]]) - zwizz_up - zwizz_down)
    return H

def ham_rk(order, state1, state2, N, t=1, U=2, mu=0):
    """
    Retourne l'element de la matrice H dans la
    base mixte "rk" ou "kr" (selon order) 
    (state1, state2) pour un systeme a N sites
    avec des parametres t, U et mu
    """

    state1_up = state1[:N]
    state1_down = state1[N:]
    n_up1 = sum(state1_up)
    n_down1 = sum(state1_down)
    state2_up = state2[:N]
    state2_down = state2[N:]
    n_up2 = sum(state2_up)
    n_down2 = sum(state2_down)

    # Cas non physique (non conservation de N_e/S_z)
    if n_up1 != n_up2 or n_down1 != n_down2:
        H = 0
        return H

    # Partie cinetique (diagonale en k)
    H_t = 0
    for i in range(N):
        if order == "rk":
            H_t += -2*t * np.cos((2*np.pi*i)/N) * (state2[i] + state2[i + N])
        elif order == "kr":
            H_t += -2*t * np.cos((2*np.pi*i)/N) * (state1[i] + state1[i + N])

    # Partie d'interaction (diagonale en r)
    H_U = 0
    for i in range(N):
        if order == "rk":
            H_U += U * state1[i] * state1[i + N]
        if order == "kr":
            H_U += U * state2[i] * state2[i + N]

    # Partie de potentiel chimique (diagonale en r et en k)
    H_mu = - mu * sum(state1)

    H = met_rk(order, state1, state2, N) * (H_t + H_U + H_mu)
    return H

def ground_energy(states, N, t=1, U=2, mu=0):  
    """
    Prend en entree un ensemble d'etats de la base mixte 
    (states) et la taille du systeme (N) avec parametres t, 
    U et mu et retourne les valeurs propres du systeme 
    (eigenvalues), les vecteurs propres en ligne 
    (eigenvectors.T), les matrices H et S (new_H et new_S), 
    ainsi qu'un indicateur de surcompletude de la base 
    d'etats mixtes (overfilled)

    *Creation des matrices element par element => diagonalisation*
    """ 
    
    # Creation des matrices vides
    dim = len(states)
    H = np.zeros((dim, dim), dtype=complex)
    S = np.zeros((dim, dim), dtype=complex)

    # Creation des matrices (triangulaires superieures) element par element
    for i in range(dim):
        for j in range(i, dim):
            if states[i].type == "r" and states[j].type == "r":    # cas <r|H|r>          
                H[i, j] = ham_r(states[i].vector, states[j].vector, N, t, U, mu)
                S[i, j] = met_r(states[i].vector, states[j].vector, N)
            elif states[i].type == "r" and states[j].type == "k":    # cas <r|H|k>
                H[i, j] = ham_rk("rk", states[i].vector, states[j].vector, N, t, U, mu)
                S[i, j] = met_rk("rk", states[i].vector, states[j].vector, N)
            elif states[i].type == "k" and states[j].type == "r":    # cas <k|H|r>
                H[i, j] = ham_rk("kr", states[i].vector, states[j].vector, N, t, U, mu)
                S[i, j] = met_rk("kr", states[i].vector, states[j].vector, N)
            elif states[i].type == "k" and states[j].type == "k":    # cas <k|H|k>
                H[i, j] = ham_k(states[i].vector, states[j].vector, N, t, U, mu)
                S[i, j] = met_k(states[i].vector, states[j].vector, N)
    
    # Utilisation de la propriete d'hermiticite des matrices
    new_H = H + H.conj().T
    np.fill_diagonal(new_H, np.diag(H).real)
    new_S = S + S.conj().T
    np.fill_diagonal(new_S, np.diag(S).real)

    # Verification de la surcompletude de la base (valeurs propres de S)
    eig_S = np.linalg.eigh(new_S)[0]

    # Au moins une valeur propre de S nulle => diagonalisation generalisee
    if np.any(np.isclose(eig_S, 0, atol=1e-6)):        
        eigenvalues, eigenvectors = gen_diagonalization(new_H, new_S)        
        overfilled = True

    # Aucune valeur propre de S nulle => diagonalisation par eigh
    else:
        eigenvalues, eigenvectors = sp.linalg.eigh(new_H, new_S)
        overfilled = False

    return eigenvalues, new_H, new_S, eigenvectors.T, overfilled



# Creation aleatoire d'une base mixte complete (pas surcomplete)

def complete_basis(S, states, N):
    """
    Prend en entree une matrice complete S et l'ensemble 
    des etats de la base mixte en r et en k pour un 
    systeme a N sites et retourne une base mixte aleatoire 
    qui n'est pas surcomplete
    """

    # Liste des indices de la base utilisee
    n = []
    for element in states:
        if element.type == "r":
            n.append(element.number())
        if element.type == "k":
            n.append(element.number())
    n = np.array(n)

    # Choix aleatoire d'une base de taille N non surcomplete
    eigvals = [0]    
    while np.isclose(eigvals[0], 0, atol=1e-6):    # test de base surcomplete
        comb = np.sort(np.random.choice(len(states), size=len(states)//2, replace=False))
        new_S = S[np.ix_(n[comb], n[comb])]
        eigvals, _ = np.linalg.eigh(new_S)

    return [states[i] for i in comb]



## Test 2 sites => 3 etats

"""
Retourne les bases mixtes du systeme a 2 sites 
avec des parametres t et U suffisant a retrouver 
l'energie fondamentale

*Une correction a la valeur de t doit etre apportee
(t = 1/2*t) en raison de l'absence de symetrie de 
translation avec le systeme a 2 sites* 
"""

etats = [5, 6, 9, 10]

N = 2
t = 1
U = 2

def test_2(etats, t=1/2, U=2):

    # Creation des etats en r et en k
    states_r = []
    states_k = []
    states = []
    for element in etats:
        ket_r = Ket(list(map(int, format(element, f'0{4}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{4}b'))), "k", str(element) + "k")
        states.append(ket_r)
        states_r.append(ket_r)
        states.append(ket_k)
        states_k.append(ket_k)

    # Energie fondamentale analytique a 2 sites
    E0 = U/2 - np.sqrt((U/2)**2 + 16*(t**2))

    # Test des combinaisons de 3 etats
    for etats in combinations(states, 3):        
        E = ground_energy(list(etats), N, t, U)[0]
        if np.isclose(E[0], E0):
            print([s.name for s in etats])
            print(round(E[0], 5))

#test_2(etats, 1/2*t, U) 



## Test N sites : sous-espaces r/k et energies fondamentales associees

"""
Diagonalisation du Hamiltonien dans la base des r
et dans la base des k et separation des etats
selon les blocs de symetrie pour un systeme a N 
sites avec des parametres t, U et mu
"""

N = 4
t = 1
U = 2
mu = 1

def block_diagonalization(N=4, t=1, U=2, mu=0):

    dim = 4**N

    ## Base r

    # Creation des etats
    r_states = np.arange(dim)
    etats_r = []
    for r in r_states:    
        ket_r = Ket(list(map(int, format(r, f'0{8}b'))), "r", str(r) + "r")
        etats_r.append(ket_r)

    # Creation de la matrice H
    H_r = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            H_r[i, j] = ham_r(etats_r[i].vector, etats_r[j].vector, N, t, U, mu)

    # Separation en blocs de symetrie
    comps_r = all_components(H_r)
    fund_r = []

    # Diagonalisation par bloc et classification des energies propres
    for c in comps_r:
        print(c)
        S_sorted = sorted(c)
        H_sub = H_r[np.ix_(S_sorted, S_sorted)]
    
        eigvals = []
        for val in np.linalg.eigvals(H_sub): 
            eigvals.append(float(round(val.real, 5)))
        for el in eigvals:
            fund_r.append(el)
        print(sorted(eigvals))

    print(sorted(fund_r))

    ## Base k

    # Creation des etats
    k_states = np.arange(dim)
    etats_k = []
    for k in k_states:   
        ket_k = Ket(list(map(int, format(k, f'0{8}b'))), "k", str(k) + "k")
        etats_k.append(ket_k)

    # Creation de la matrice H
    H_k = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            H_k[i, j] = ham_k(etats_k[i].vector, etats_k[j].vector, N, t, U, mu)

    # Separation en blocs de symetrie
    comps_k = all_components(H_k)
    fund_k = []

    # Diagonalisation par bloc et classification des energies propres
    for c in comps_k:
        print(c)
        S_sorted = sorted(c)
        H_sub = H_k[np.ix_(S_sorted, S_sorted)]

        eigvals = []
        for val in np.linalg.eigvals(H_sub): 
            eigvals.append(float(round(val.real, 5)))
        for el in eigvals:
            fund_k.append(el)
        print(sorted((eigvals)))

    print(sorted(fund_k))

    return

#block_diagonalization(N, t, U, mu)



## Test 4-sites : base mixte de 2 a 8 etats (tests si mieux que k pur)

N = 4
t = 1
U = 2
n = 150

def optimized_fund_4(states, H, S, N=4, t=1, U=2):
    """
    Prend en entree tous les etats de la base
    mixte et reduit la base en retirant:
    - aleatoirement un etat de la base si son 
    retrait ne change pas la valeur de l'energie 
    propre fondamentale (quand la taille de la 
    base est grande)
    - dans l'ordre un etat de la base si son
    retrait ne change pas la valeur de l'energie 
    propre fondamentale (quand la taille de la 
    base est petite)
    """   

    # Energie fondamentale analytique a 4 sites
    beta = np.arccos((4*(t**2)*U)/(16/3*(t**2) + 1/3*(U**2))**(3/2))
    E0 = U - (2/np.sqrt(3))*np.sqrt(16*(t**2) + U**2)*np.cos(beta/3)

    # Tant qu'il est possible de retirer un etat
    while True:
        if len(states) > 30:    # base de grande taille
            aleatoire = True
        else:    # base de petite taille
            aleatoire = False

        # Recherche d'un etat a retirer
        removed = False
        for i in range(len(states)):
            new_states = states.copy()
            idx = np.random.randint(len(states)) if aleatoire else i
            new_states.pop(idx)
            E = optimized_ground_energy(new_states, H, S, N, t, U)[0]

            # Verification de l'impact sur le fondamental
            if np.isclose(E[0], E0):
                states = new_states
                removed = True
                break

        # Aucun etat ne peut etre retire
        if not removed:
            break

    # Ajout de la base mixte si pertinente
    if len(states) <= 10 and states not in ensembles:
        ensembles.append(states)
        tailles.append(len(states))

    return states

def test_optimized_fund_4(n, N=4, t=1, U=2):
    """
    Cree tous les etats de la base mixte
    et reduit n fois la base complete
    dans le but de retrouver la majorite
    des ensembles d'etats de la base mixte
    de taille inferieure a 11 permettant
    de retrouver l'energie fondamentale
    Retourne ces ensembles d'etats et 
    leur taille
    """

    # Creation de tous les etats de la base mixte
    states = []
    etats = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]
    for element in etats:
        ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")
        states.append(ket_r)
        states.append(ket_k)

    H, S = get_H_S(N, t, U)
    global ensembles
    global tailles
    ensembles = []
    tailles = []

    # Reduction de la base (n fois)    
    for i in range(n):
        optimized_fund(states, H, S, N, t, U)

    # Affichage des resultats
    for i in range(len(ensembles)):
        print(f"\n{ensembles[i]} \n {tailles[i]}")

    return ensembles, tailles

#test_optimized_fund_4(n, N, t, U)

def minor_polynom(E0, eigvals, eigvec):
    """
    Prend en entree la plus petite energie
    propre d'une base (E0) ainsi que les valeurs
    propres (eigvals) et le i-eme vecteur propre
    (eigvec). Construit le polynome caracteristique
    associe au retrait de l'etat correspondant
    a la i-eme dimension et l'evalue en E0 
    (dans l'espoir d'y trouver encore un
    changement de signe)
    """

    # Creation du polynome/evaluation du polynome en E0
    val = 0
    for k in range(len(eigvals)):
        term = abs(eigvec[k])**2 * prod((E0 - eigvals[j]) for j in range(len(eigvals)) if j != k)
        val += term

    return val

def plus_optimized_fund_4(N=4, t=1, U=2, eps=1e-10):
    """
    Cree tous les etats de la base mixte et 
    reduit la base en retirant chaque direction 
    possible

    *Utilisation d'une methode de reduction basee
    sur le polynome caracteristique*

    *Attention: pas optimise => tres long*
    """ 

    # Creation de tous les etats de la base mixte
    states = []
    etats = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]
    for element in etats:
        ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")
        states.append(ket_r)
        states.append(ket_k)

    H, S = get_H_S(N, t, U)
    plus_ensembles = []
    plus_tailles = []

    # Energie fondamentale analytique a 4 sites
    beta = np.arccos((4*(t**2)*U)/(16/3*(t**2) + 1/3*(U**2))**(3/2))
    E0 = U - (2/np.sqrt(3))*np.sqrt(16*(t**2) + U**2)*np.cos(beta/3)

    # Liste des bases à explorer
    states_to_check = [states]

    while states_to_check:

        # Prend une base à explorer
        current_states = states_to_check.pop()

        # Diagonalisation de la base
        eigvals, _, _, eigvecs, _ = optimized_ground_energy(current_states, H, S, N, t, U)

        index_to_flush = []

        # Recherche de tous les états qui peuvent être retirés
        for i in range(len(eigvals)):
            left = minor_polynom(E0 - eps, eigvals, eigvecs[i, :])
            right = minor_polynom(E0 + eps, eigvals, eigvecs[i, :]) 
            if left*right < 0:
                index_to_flush.append(i)

        # Aucun etat ne peut etre retire
        if not index_to_flush:

            # Ajout de la base mixte si pertinente
            if len(current_states) <= 10 and current_states not in plus_ensembles:
                plus_ensembles.append(current_states)
                plus_tailles.append(len(current_states))

            continue

        # Pour chaque etat qui peut etre retire
        for index in index_to_flush:
            new_states = current_states.copy()
            new_states.pop(index)
            states_to_check.append(new_states)

    # Affichage des resultats
    for i in range(len(plus_ensembles)):
        print(f"\n{plus_ensembles[i]} \n {plus_tailles[i]}")

    return plus_ensembles, plus_tailles

#plus_optimized_fund_4(N, t, U)

def best_k(states_k, N=4, t=1, U=2):
    """
    Prend en entree des etats (vecteurs) de la
    base k et trouve les energies propres minimales 
    retrouvees avec un nombre d'etats variant
    de 4 a 10 pour un systeme a N sites avec
    des parametres t et U
    """

    # Energie minimale pour i nombre d'etats
    E = []
    for i in range(4, 11):
        E_i = []
        for etats_k in combinations(states_k, i):
            E0 = ground_energy(list(etats_k), N, t, U)[0][0]
            E_i.append(E0)
        E.append(min(E_i))
    return E

def test_4(N=4, t=1, U=2):
    """
    Cree tous les etats de la base mixte et teste
    chaque combinaison d'etats de taille de 2 a 8
    composee d'etats k du bloc N_e=4, S_z=0, K=pi 
    (fondamental) et d'etats r du bloc N_e=4 S_z=0; 
    compare l'energie propre minimale obtenue par 
    chaque combinaison a l'energie fondamentale
    exacte et a la plus basse energie propre obtenue
    avec ce nombre d'etats (mais en base k)
    Verifie egalement la surcompletude des bases testees

    *Peut etre adaptee facilement pour tester avec
    des combinaisons de blocs differents*
    """

    # Creation des etats de la base mixte
    etats = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]
    etats1 = [51, 60, 90, 102, 105, 150, 153, 165, 195, 204]
    etats2 = [53, 83, 92, 106, 154, 166, 169, 197]
    etats3 = [54, 57, 85, 99, 108, 147, 156,  170, 198, 201]
    etats4 = [58, 86, 89, 101, 149, 163, 172, 202]
    states_r1 = []
    states_r2 = []
    states_r3 = []
    states_r4 = []
    states_k1 = []
    states_k2 = []
    states_k3 = []
    states_k4 = []
    states_r = []
    states_k = []
    ensemble_minimal = []    # ensemble de 6 etats => E0
    states = []
    for element in etats1:
        ket_r = Ket(list(map(int, format(element, f'0{2*N}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{2*N}b'))), "k", str(element) + "k")
        states_r1.append(ket_r)
        states_r.append(ket_r)
        states_k1.append(ket_k)
        states_k.append(ket_k)
        if element in [90, 165]:
            ensemble_minimal.append(ket_r)
            ensemble_minimal.append(ket_k)
        elif element in [153, 204]:
            ensemble_minimal.append(ket_k)
    for element in etats2:
        ket_r = Ket(list(map(int, format(element, f'0{2*N}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{2*N}b'))), "k", str(element) + "k")
        states_r2.append(ket_r)
        states_r.append(ket_r)
        states_k2.append(ket_k)
        states_k.append(ket_k)
    for element in etats3:
        ket_r = Ket(list(map(int, format(element, f'0{2*N}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{2*N}b'))), "k", str(element) + "k")
        states_r3.append(ket_r)
        states_r.append(ket_r)
        states_k3.append(ket_k)
        states_k.append(ket_k)
    for element in etats4:
        ket_r = Ket(list(map(int, format(element, f'0{2*N}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")
        states_r4.append(ket_r)
        states_r.append(ket_r)
        states_k4.append(ket_k)
        states_k.append(ket_k)
    for element in etats:
        ket_r = Ket(list(map(int, format(element, f'0{2*N}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{2*N}b'))), "k", str(element) + "k")
        states.append(ket_r)
        states.append(ket_k)

    # Plus basses energies obtenues avec des etats k
    result = best_k(states_k1)
    best = result[:-1]
    fund = result[-1]

    H, S = get_H_S(N, t, U)

    # Test de chaque combinaison d'etats mixtes
    for i in range(1, 8):    # nombre d'etats en r
        for j in range(1, 8):    # nombre d'etats en k
            if i + j <= 8:    
                print(f"\033[1;36m{i} etats en r et {j} etats en k\n\033[0m")

                # Ajout des etats en r
                for etats_r in combinations(states_r, i):
                    states = []
                    for e in etats_r:
                        states.append(e)
    
                    # Ajout des etats en k
                    for etats_k in combinations(states_k1, j):
                        new_states = states.copy()
                        for e in etats_k:
                            new_states.append(e)

                        # Diagonalisation dans la base mixte
                        E0, _, _, _, overfilled = optimized_ground_energy(new_states, H, S, 4, t, U)

                        # Energie fondamentale a 6 etats
                        if set(new_states) == set(ensemble_minimal):
                            print("Energie fondamentale\n")
                            print([s.name for s in new_states])
                            print(round(E0, 5))
                            print("\n")

                        # Retrait des doublons de E0 a 6 etats
                        elif set(ensemble_minimal).issubset(new_states):
                            continue

                        # Verification de surcompletude
                        elif overfilled:
                            print("Base surcomplete\n")
                            print([s.name for s in new_states])
                            print("\n")
                        
                        # Energie fondamentale
                        elif np.isclose(min(energies), fund):  
                            print("Energie fondamentale\n")
                            print([s.name for s in new_states])
                            print(round(fund, 5))
                            print("\n")

                        # Meilleur que dans la base k
                        else:                  
                            for ind, E in zip(range(9, 3, -1), reversed(best)):
                                if E0 < E and i + j < ind:
                                    print(f"Mieux que {ind} etats k\n")
                                    print([s.name for s in new_states])
                                    print(round(E0, 5))
                                    print("\n")
                                    break

    return

#test_4(N, t, U)



## Test N sites : energie fondamentale et etat fondamental

"""
Creation d'une base mixte composee des etats en r
(r) et des etats en k (k) pour un systeme N sites 
avec les parametres t, U et mu et recherche de la plus 
petite valeur propre (energie fondamentale)
Retourne l'energie fondamentale, les matrices
H et S, l'etat fondamental, le statut de la base
mixte utilisee (surcomplete ou non) ainsi que
le temps de calcul
"""

r = [90, 165]
k = [90, 153, 165, 204]

N = 4
t = 1
U = 2
mu = 0


def mixte(r, k, N, t=1, U=2, mu=0):
    
    debut = time.perf_counter()
    
    # Creation des etats r et k
    states = []
    for r in r:
        ket_r = Ket(list(map(int, format(r, f'0{2*N}b'))), "r", str(r) + "r")
        states.append(ket_r)
    for k in k:
        ket_k = Ket(list(map(int, format(k, f'0{2*N}b'))), "k", str(k) + "k")
        states.append(ket_k)

    # Diagonalisation
    E, H, S, omega, overfilled = ground_energy(states, N, t, U, mu)

    fin = time.perf_counter()

    # Affichage des resultats
    print(states)
    if overfilled == True:
        print("\n")
        print("BASE SURCOMPLETE")
    print("\n")
    print("Energie fondamentale : ", round(min(E), 5))
    print("\n")
    print("H = \n", H)
    print("\n")
    print("S = \n", S)
    print("\n")
    print("Etat fondamental : ", omega[0])
    print("\n")
    print(f"Temps d'exécution total : {fin - debut:.6f} s")
    return E[0], H, S, omega[0]

#mixte(r, k, N, t, U, mu)



## Test 6-sites : base mixte

N = 6
t = 1
U = 2
n = 150
n1 = 55

def states_creation(N):
    """
    Cree tous les entiers dont la forme binaire
    represente un etat du bloc a demi-rempli du 
    systeme a N sites
    """

    # Creation des vecteurs (etats)
    states = []
    for up in combinations(range(N), N//2):
        for down in combinations(range(N), N//2):
            vector = [0] * (2 * N)
            for i in up:
                vector[i] = 1
            for i in down:
                vector[N + i] = 1

            # Conversion de l'etat en entier
            number = int("".join(map(str, vector)), 2)

            states.append(number)
    return states

def test_6(n1, N=6, t=1, U=2):
    """
    Retourne les etats de la base mixte (n) du 
    systeme a 6 sites avec des parametres t et U 
    suffisant a retrouver l'energie fondamentale
    """

    # Creation de tous les etats de la base mixte
    states = []
    etats = states_creation(N)
    for element in etats:
        ket_r = Ket(list(map(int, format(element, f'0{2*N}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{2*N}b'))), "k", str(element) + "k")
        states.append(ket_r)
        states.append(ket_k)

    # Energie fondamentale (numerique) a 6 sites
    E0 = -5.4094568451    # pour t=1 et U=2

    H, S = get_H_S(N, t, U) 

    # Test des combinaisons de 2 etats mixtes
    for i in range(math.comb(400, n1)):    
        etats = random.sample(states, n1)

        # Diagonalisation       
        E = optimized_ground_energy(list(etats), H, S, N, t, U)[0][0]

        if np.isclose(E, E0):
            print([s.name for s in etats])
            print(round(E0, 5))
    return

#test_6(n1, N, t, U)

def optimized_fund_6(states, H, S, N=6, t=1, U=2):
    """
    Prend en entree tous les etats de la base
    mixte et reduit la base en retirant:
    - aleatoirement un etat de la base si son 
    retrait ne change pas la valeur de l'energie 
    propre fondamentale (quand la taille de la 
    base est grande)
    - dans l'ordre un etat de la base si son
    retrait ne change pas la valeur de l'energie 
    propre fondamentale (quand la taille de la 
    base est petite)
    """   

    # Energie fondamentale (numerique) a 6 sites
    E0 = -5.4094568451    # pour t=1 et U=2

    # Tant qu'il est possible de retirer un etat
    while True:
        if len(states) > 90:    # base de grande taille
            aleatoire = True
        else:    # base de petite taille
            aleatoire = False

        # Recherche d'un etat a retirer
        removed = False
        for i in range(len(states)):
            new_states = states.copy()
            idx = np.random.randint(len(states)) if aleatoire else i
            new_states.pop(idx)
            E = optimized_ground_energy(new_states, H, S, N, t, U)[0]

            # Verification de l'impact sur le fondamental
            if np.isclose(E[0], E0, atol=1e-3):
                states = new_states
                removed = True
                break

        # Aucun etat ne peut etre retire
        if not removed:
            break

    # Ajout de la base mixte si pertinente
    if len(states) <= 70 and states not in ensembles_6:
        ensembles_6.append(states)
        tailles_6.append(len(states))

    return states

def test_optimized_fund_6(n, N=6, t=1, U=2):
    """
    Cree tous les etats de la base mixte
    et reduit n fois la base complete
    dans le but de retrouver la majorite
    des ensembles d'etats de la base mixte
    de taille inferieure a 71 permettant
    de retrouver l'energie fondamentale
    Retourne ces ensembles d'etats et 
    leur taille
    """

    # Creation de tous les etats de la base mixte
    states = []
    etats = states_creation(N)
    for element in etats:
        ket_r = Ket(list(map(int, format(element, f'0{2*N}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{2*N}b'))), "k", str(element) + "k")
        states.append(ket_r)
        states.append(ket_k)

    H, S = get_H_S(N, t, U)
    global ensembles_6
    global tailles_6
    ensembles_6 = []
    tailles_6 = []

    # Reduction de la base (n fois)    
    for i in range(n):
        optimized_fund(states, H, S, N, t, U)

    # Affichage des resultats
    for i in range(len(ensembles)):
        print(f"\n{ensembles_6[i]} \n {tailles_6[i]}")

    return ensembles_6, tailles_6

#test_optimized_fund_6(n, N, t, U)

def plus_optimized_fund_6(N=6, t=1, U=2, eps=1e-10):
    """
    Cree tous les etats de la base mixte et 
    reduit la base en retirant chaque direction 
    possible

    *Utilisation d'une methode de reduction basee
    sur le polynome caracteristique*

    *Attention: pas optimise => inutilisable pour l'instant*
    """ 

    # Creation de tous les etats de la base mixte
    states = []
    etats = states_creation(N)
    for element in etats:
        ket_r = Ket(list(map(int, format(element, f'0{2*N}b'))), "r", str(element) + "r")
        ket_k = Ket(list(map(int, format(element, f'0{2*N}b'))), "k", str(element) + "k")
        states.append(ket_r)
        states.append(ket_k)

    H, S = get_H_S(N, t, U)
    plus_ensembles = []
    plus_tailles = []

    # Energie fondamentale (numerique) a 6 sites
    E0 = -5.4094568451    # pour t=1 et U=2

    # Liste des bases à explorer
    states_to_check = [states]

    while states_to_check:

        # Prend une base à explorer
        current_states = states_to_check.pop()

        # Diagonalisation de la base
        eigvals, _, _, eigvecs, _ = optimized_ground_energy(current_states, H, S, N, t, U)

        index_to_flush = []

        # Recherche de tous les états qui peuvent être retirés
        for i in range(len(eigvals)):
            left = minor_polynom(E0 - eps, eigvals, eigvecs[i, :])
            right = minor_polynom(E0 + eps, eigvals, eigvecs[i, :]) 
            if left*right < 0:
                index_to_flush.append(i)

        # Aucun etat ne peut etre retire
        if not index_to_flush:

            # Ajout de la base mixte si pertinente
            if len(current_states) <= 70 and current_states not in plus_ensembles:
                plus_ensembles.append(current_states)
                plus_tailles.append(len(current_states))

            continue

        # Pour chaque etat qui peut etre retire
        for index in index_to_flush:
            new_states = current_states.copy()
            new_states.pop(index)
            states_to_check.append(new_states)

    # Affichage des resultats
    for i in range(len(plus_ensembles)):
        print(f"\n{plus_ensembles[i]} \n {plus_tailles[i]}")

    return plus_ensembles, plus_tailles

#plus_optimized_fund_6(N, t, U)



## Test etat mixte 2 sites

"""
Retourne les etats mixtes (n) du systeme a 2 sites 
avec des parametres t et U suffisant a retrouver 
l'energie fondamentale

*Une correction a la valeur de t doit etre apportee
(t = 1/2*t) en raison de l'absence de symetrie de 
translation avec le systeme a 2 sites* 
"""

N = 2
t = 1
U = 2
n = 2

def m_test_2(n=2, N=2, t=1/2, U=2):

    # Creation des etats mixtes
    etats = np.arange(4**(2*N))
    states = []
    compteur = 0
    for element in etats:
        lst = list(map(int, format(element, f'0{4*N}b')))
        if sum(lst[:2*N]) != 1 or sum(lst[2*N:]) != 1:
            continue
        ket = Ket(lst, None, str(element), compteur=compteur)
        states.append(ket)
        compteur += 1    # indice matriciel

    # Energie fondamentale analytique a 2 sites
    E0 = U/2 - np.sqrt((U/2)**2 + 16*(t**2))

    H, S = m_get_H_S(N, t, U) 

    # Test des combinaisons de 2 etats mixtes
    for etats in combinations(states, n):

        # Diagonalisation       
        E = m_optimized_ground_energy(list(etats), H, S, N, t, U)[0][0]
        
        if np.isclose(E, E0):
            print([s.name for s in etats])
            print(round(E0, 5))
    return

#m_test_2(n, N, t/2, U)



## Test etat mixte 4 sites

"""
Retourne les etats mixtes (n) du systeme a 4 sites 
avec des parametres t et U suffisant a retrouver 
l'energie fondamentale
"""

N = 4
t = 1
U = 2
n = 6

def m_test_4(n, N=4, t=1, U=2):

    # Creation des etats mixtes
    etats = np.arange(4**(2*N))
    states = []
    for element in etats:
        lst = list(map(int, format(element, f'0{4*N}b')))
        if sum(lst[:2*N]) != 2 or sum(lst[2*N:]) != 2:
            continue
        ket = Ket(lst, None, str(element))
        states.append(ket)
    
    # Energie fondamentale analytique a 4 sites
    beta = np.arccos((4*(t**2)*U)/(16/3*(t**2) + 1/3*(U**2))**(3/2))
    E0 = U - (2/np.sqrt(3))*np.sqrt(16*(t**2) + U**2)*np.cos(beta/3)

    H, S = m_get_H_S(N, t, U)

    # Test des combinaisons de n etats mixtes
    for i in range(math.comb(784, n)):    
        etats = random.sample(states, n)

        # Diagonalisation   
        E = m_optimized_ground_energy(list(etats), H, S, N, t, U)[0]

        if np.isclose(E, E0):
            print([s.name for s in etats])
            print(round(E0, 5))
            print("\n")
    return

#m_test_4(n, N, t, U)



## Test etats mixtes N sites : energie fondamentale et etat fondamental

"""
Creation d'une base d'etats mixtes pour un systeme 
N sites avec les parametres t et U et recherche de 
la plus petite valeur propre (energie fondamentale)
Retourne l'energie fondamentale, les matrices
H et S, l'etat fondamental, le statut de la base
mixte utilisee (surcomplete ou non) ainsi que
le temps de calcul
"""

etats = [576, 1026, 4100, 320]

N = 4
t = 1
U = 2

def etats_mixtes(etats, N, t=1, U=2):
    
    debut = time.perf_counter()

    # Creation de tous les etats mixtes
    els = np.arange(4**(2*N))
    sts = []
    compteur = 0
    for el in els:
        lst = list(map(int, format(el, f'0{4*N}b')))
        if sum(lst[:2*N]) != 1 or sum(lst[2*N:]) != 1:
            continue
        ket = Ket(lst, None, str(el), compteur=compteur)
        sts.append(ket)
        compteur += 1    # indice matriciel

    # Creation des etats mixtes
    etats_set = set(etats)
    states = [st for st in sts if st.number() in etats_set]
    if len(states) != len(etats):
        return print("etats mixtes non physiques")
    
    H, S = m_get_H_S(N, t, U)

    # Diagonalisation
    E, H, S, omega, overfilled = m_optimized_ground_energy(states, H, S, N, t, U)

    fin = time.perf_counter()

    # Affichage des resultats
    print(states)
    if overfilled == True:
        print("\n")
        print("BASE SURCOMPLETE")
    print("\n")
    print("Energie fondamentale : ", round(min(E), 5))
    print("\n")
    print("H = \n", H)
    print("\n")
    print("S = \n", S)
    print("\n")
    print("Etat fondamental : ", omega[0])
    print("\n")
    print(f"Temps d'exécution total : {fin - debut:.6f} s")
    return E[0], H, S, omega[0]

#etats_mixtes(etats, N, t, U)



## Fonction de Green non orthonormee (construction de N sous-espaces excites mu)

"""
Prend en entree une base mixte composee d'etats r
et d'etats k et retourne la densite d'etats du 
systeme a demi-rempli a N sites avec des parametres 
t, U et mu pour une base non orthonormee : les
etats de la base de depart sont excites par un 
electron/trou de spin "spin" et de type "basis"

*Les sous-espaces excites sont crees independamment
par chacun des mu (chaque excitation de site/k)*
"""

# Etats de la base de depart

#r = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]
#k = []

#r = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]
#k = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]

r = [90, 165]
k = [90, 165, 153, 204]

#r = []
#k = [51, 60, 90, 102, 105, 150, 153, 165, 195, 204]

#r = [51, 102, 153, 204]
#k = [51, 102, 153, 204]

#r = [51, 54, 57, 58, 83, 85, 86, 89, 90, 92, 99, 101, 105, 106, 108, 147, 149, 154, 156, 163, 165, 166, 169, 172, 195, 197, 201]
#k = [53, 60, 102, 150, 153, 170, 198, 202, 204]

#r = [85, 170]
#k = [51, 60, 90, 153, 165, 195, 204]

#r = [51, 60, 90, 165, 195, 204]
#k = [85, 153, 170, 204]

#r = [85, 170]
#k = [90, 102, 105, 150, 153, 165, 204]

#r = [51, 60, 153, 204]
#k = []

# Parametres du systeme
N = 4
t = 1
U = 2
mu = U/2

# Parametres de l'excitation
spin = 0        # spin up --> 0  /  spin down --> 1
basis = "k"

def etats_decomposes(N):
    """
    Retourne tous les vecteurs/etats comportant
    N/2 electrons up et N/2 electrons down
    => Transformee de Fourier r -> k ou k -> r
    """
  
    # Creation de tous les etats up/down a N/2 electrons 
    ensemble = []
    for positions in combinations(range(N), N//2):
        liste = [0] * N
        for i in positions:
            liste[i] = 1
        ensemble.append(liste)

    # Creation de tous les etats up + down
    resultat = []
    for a in ensemble:
        for b in ensemble:
            resultat.append(a + b)
    return resultat

def Ht_mu(state1, state2, N, mu, t, excitation):
    """
    Retourne le terme de saut du hamiltonien excite en r
    de (state1, state2) pour un systeme a N sites avec 
    un parametre t dont:
    - le site d'arrivee est mu (excitation "e")
    - le site de depart est mu (excitation "h")
    """

    # Differences entre state1 et state2
    diff = [b - a for a, b in zip(state1, state2)]
    if diff.count(1) != 1 or diff.count(-1) != 1:    # cas de plusieurs sauts
        return 0
    i = diff.index(1)
    j = diff.index(-1)
    
    # Verification des sites de depart/arrivee
    if excitation == "e":
        if j != mu:
            return 0    
    elif excitation == "h":
        if i != mu:
            return 0

    # Signes de permutation
    if abs(i - j) == 1:    # saut conventionnel
        exp = 0
    elif abs(i - j) == N - 1:    # saut par symetrie de translation
        exp = N // 2 - 1
    else:    # saut de plusieurs sites
        return 0

    return t * (-1) ** exp

def S_plus_r(states, N, mu, spin):
    """
    Retourne la metrique excitee en electrons
    pour une base de depart states, un systeme
    a N sites et une excitation localisee en mu
    de spin "spin"
    """

    S = np.load(f"S_{N}.npy")
    
    # Creation de la matrice vide
    dim = len(states)
    S_plus = np.zeros((dim, dim), dtype=complex)

    # Creation de la matrice element par element
    for i in range(dim):
        for j in range(dim):
            if states[i].type == "r" and states[j].type == "r":    # cas <r|r>
                S_plus[i, j] = int(i == j) * int(states[i].vector[mu + N*spin] == 0)
            elif states[i].type == "r" and states[j].type == "k":    # cas <r|k>
                S_plus[i, j] = int(states[i].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]
            elif states[i].type == "k" and states[j].type == "r":    #cas <k|r>
                S_plus[i, j] = int(states[j].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]
            elif states[i].type == "k" and states[j].type == "k":    # cas <k|k>
                
                # Decomposition de l'etat k en etats r
                S_plus[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2)]
                    s2 = S[int(''.join(map(str, etats[l])), 2), states[j].number()]
                    S_plus[i, j] += int(etats[l][mu + N*spin] == 0) * s1 * s2

    return S_plus

def H_plus_r(states, N, mu, spin, t, U, mu1):
    """
    Retourne la matrice H excitee en electrons
    pour une base de depart states, un systeme
    a N sites avec des parametres t, U et mu1
    et une excitation localisee en mu de spin "spin"
    """   

    S = np.load(f"S_{N}.npy")

    # Creation de la matrice vide
    dim = len(states)
    H_plus = np.zeros((dim, dim), dtype=complex)
    
    # Creation de la matrice element par element
    for i in range(dim):
        for j in range(dim):
            if states[i].type == "r" and states[j].type == "r":    # cas <r|H|r>
                delta = int(i == j)
                term1 = Ht_mu(states[i].vector, states[j].vector, N, mu + N*spin, t, "e")
                term2 = U * int(states[j].vector[mu + N*int(spin == 0)] == 1) * delta
                term3 = -mu1 * delta
                term4 = ham_r(states[i].vector, states[j].vector, N, t, U, mu1)
                H_plus[i, j] = term1 + int(states[j].vector[mu + N*spin] == 0) * (term2 + term3 + term4)
            elif states[i].type == "r" and states[j].type == "k":    # cas <r|H|k>
                
                # Decomposition de l'etat k en r
                H_plus[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    delta = int(np.array_equal(states[i].vector, etats[l]))
                    term1 = Ht_mu(states[i].vector, etats[l], N, mu + N*spin, t, "e")
                    term2 = U * int(etats[l][mu + N*int(spin == 0)] == 1) * delta
                    term3 = -mu1 * delta
                    term4 = ham_r(states[i].vector, etats[l], N, t, U, mu1)
                    s = S[int(''.join(map(str, etats[l])), 2), states[j].number()]
                    H_plus[i, j] += s * (term1 + int(etats[l][mu + N*spin] == 0) * (term2 + term3 + term4))

            elif states[i].type == "k" and states[j].type == "r":    # cas <k|H|r>

                # Decomposition de l'etat k en r
                H_plus[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    delta = int(np.array_equal(etats[l], states[j].vector))
                    term1 = Ht_mu(etats[l], states[j].vector, N, mu + N*spin, t, "e")
                    term2 = U * int(states[j].vector[mu + N*int(spin == 0)] == 1) * delta
                    term3 = -mu1 * delta
                    term4 = ham_r(etats[l], states[j].vector, N, t, U, mu1)
                    s = S[states[i].number(), int(''.join(map(str, etats[l])), 2)]
                    H_plus[i, j] += s * (term1 + int(states[j].vector[mu + N*spin] == 0) * (term2 + term3 + term4))

            elif states[i].type == "k" and states[j].type == "k":    # cas <k|H|k>

                # Decomposition des deux etats k en r
                H_plus[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    for m in range(len(etats)):
                        delta = int(np.array_equal(etats[l], etats[m]))
                        term1 = Ht_mu(etats[l], etats[m], N, mu + N*spin, t, "e")
                        term2 = U * int(etats[m][mu + N*int(spin == 0)] == 1) * delta
                        term3 = -mu1 * delta
                        term4 = ham_r(etats[l], etats[m], N, t, U, mu1)
                        s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2)]
                        s2 = S[int(''.join(map(str, etats[m])), 2), states[j].number()]
                        H_plus[i, j] += s1 * s2 * (term1 + int(etats[m][mu + N*spin] == 0) * (term2 + term3 + term4)) 

    return H_plus

def S_moins_r(states, N, mu, spin):
    """
    Retourne la metrique excitee en trous
    pour une base de depart states, un systeme
    a N sites et une excitation localisee en mu
    de spin "spin"
    """

    S = np.load(f"S_{N}.npy")

    # Creation de la matrice vide
    dim = len(states)
    S_moins = np.zeros((dim, dim), dtype=complex)

    # Creation de la matrice element par element
    for i in range(dim):
        for j in range(dim):
            if states[i].type == "r" and states[j].type == "r":    # cas <r|r>
                S_moins[i, j] = int(i == j) * int(states[i].vector[mu + N*spin] == 1)
            elif states[i].type == "r" and states[j].type == "k":    # cas <r|k>
                S_moins[i, j] = int(states[i].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]
            elif states[i].type == "k" and states[j].type == "r":    # cas <k|r>
                S_moins[i, j] = int(states[j].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]
            elif states[i].type == "k" and states[j].type == "k":    # cas <k|k>

                # Decomposition de l'etat k en etats r
                S_moins[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2)]
                    s2 = S[int(''.join(map(str, etats[l])), 2), states[j].number()]
                    S_moins[i, j] += int(etats[l][mu + N*spin] == 1) * s1 * s2

    return S_moins

def H_moins_r(states, N, mu, spin, t, U, mu1):
    """
    Retourne la matrice H excitee en trous
    pour une base de depart states, un systeme
    a N sites avec des parametres t, U et mu1
    et une excitation localisee en mu de spin "spin"
    """

    S = np.load(f"S_{N}.npy")

    # Creation de la matrice vide
    dim = len(states)
    H_moins = np.zeros((dim, dim), dtype=complex)

    # Creation de la matrice element par element
    for i in range(dim):
        for j in range(dim):
            if states[i].type == "r" and states[j].type == "r":    # cas <r|H|r>
                delta = int(i == j)
                term1 = Ht_mu(states[i].vector, states[j].vector, N, mu + N*spin, t, "h")
                term2 = -U * int(states[j].vector[mu + N*int(spin == 0)] == 1) * delta
                term3 = mu1 * delta
                term4 = ham_r(states[i].vector, states[j].vector, N, t, U, mu1)
                H_moins[i, j] = term1 + int(states[j].vector[mu + N*spin] == 1) * (term2 + term3 + term4)
            elif states[i].type == "r" and states[j].type == "k":    # cas <r|H|k>

                # Decomposition de l'etat k en etats r
                H_moins[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    delta = int(np.array_equal(states[i].vector, etats[l]))
                    term1 = Ht_mu(states[i].vector, etats[l], N, mu + N*spin, t, "h")
                    term2 = -U * int(etats[l][mu + N*int(spin == 0)] == 1) * delta
                    term3 = mu1 * delta
                    term4 = ham_r(states[i].vector, etats[l], N, t, U, mu1)
                    s = S[int(''.join(map(str, etats[l])), 2), states[j].number()]
                    H_moins[i, j] += s * (term1 + int(etats[l][mu + N*spin] == 1) * (term2 + term3 + term4))

            elif states[i].type == "k" and states[j].type == "r":    # cas <k|H|r>

                # Decomposition de l'etat k en etats r
                H_moins[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    delta = int(np.array_equal(etats[l], states[j].vector))
                    term1 = Ht_mu(etats[l], states[j].vector, N, mu + N*spin, t, "h")
                    term2 = -U * int(states[j].vector[mu + N*int(spin == 0)] == 1) * delta
                    term3 = mu1 * delta
                    term4 = ham_r(etats[l], states[j].vector, N, t, U, mu1)
                    s = S[states[i].number(), int(''.join(map(str, etats[l])), 2)]
                    H_moins[i, j] += s * (term1 + int(states[j].vector[mu + N*spin] == 1) * (term2 + term3 + term4))

            elif states[i].type == "k" and states[j].type == "k":     # cas <k|H|k>

                # Decomposition des etats k en etats r
                H_moins[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    for m in range(len(etats)):
                        delta = int(np.array_equal(etats[l], etats[m]))
                        term1 = Ht_mu(etats[l], etats[m], N, mu + N*spin, t, "h")
                        term2 = -U * int(etats[m][mu + N*int(spin == 0)] == 1) * delta
                        term3 = mu1 * delta
                        term4 = ham_r(etats[l], etats[m], N, t, U, mu1)
                        s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2)]
                        s2 = S[int(''.join(map(str, etats[m])), 2), states[j].number()]
                        H_moins[i, j] += s1 * s2 * (term1 + int(etats[m][mu + N*spin] == 1) * (term2 + term3 + term4)) 

    return H_moins



def HU_mu(state1, state2, N, mu, spin, U, excitation):
    """
    Retourne le terme de d'interaction du hamiltonien 
    excite en k de (state1, state2) pour un systeme a 
    N sites avec un parametre U dont:
    - le site d'arrivee en "spin" est mu (excitation "e")
    - le site de depart en "spin" est mu (excitation "h")
    """
    
    # Differences entre state1 et state2
    diff = [b - a for a, b in zip(state1, state2)]
    if diff[:N].count(1) != 1 or diff[:N].count(-1) != 1 or diff[N:].count(1) != 1 or diff[N:].count(-1) != 1:    # cas de sauts non physiques
        return 0
    i = diff[:N].index(1)
    j = diff[:N].index(-1)
    k = diff[N:].index(1) + N
    l = diff[N:].index(-1) + N

    state_up = state2[:N]
    state_down = state2[N:]

    # Verification des sites de depart/arrivee
    if excitation == "e":
        if spin == 0 and j != mu:
            return 0
        elif spin == 1 and l != mu + N:
            return 0
    elif excitation == "h":
        if spin == 0 and i != mu:
            return 0
        elif spin == 1 and k != mu + N:
            return 0

    # Cas de non conservation du momentum
    if (k-l) % N != (j-i) % N:
        return 0
    
    # Signe de permutation
    exp = 0
    if i < j:    # saut up vers la droite
        exp += sum(state_up[:j]) + sum(state_up[:i]) - 1
    elif i > j:    # saut up vers la gauche
        exp += sum(state_up[:j]) + sum(state_up[:i])
    if k < l:    # saut down vers la droite
        exp += sum(state_down[:l-N]) + sum(state_down[:k-N]) - 1
    elif k > l:    # saut down vers la gauche
        exp += sum(state_down[:l-N]) + sum(state_down[:k-N])

    return -U/N * (-1) ** exp

def S_plus_k(states, N, mu, spin):
    """
    Retourne la metrique excitee en electrons
    pour une base de depart states, un systeme
    a N sites et une excitation delocalisee en mu
    de spin "spin"
    """

    S = np.load(f"S_{N}.npy")

    # Creation de la matrice vide
    dim = len(states)
    S_plus = np.zeros((dim, dim), dtype=complex)

    # Creation de la matrice element par element
    for i in range(dim):
        for j in range(dim):
            if states[i].type == "r" and states[j].type == "r":    # cas <r|r>

                # Decomposition de l'etat r en etats k
                S_plus[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2) + 4**N]
                    s2 = S[int(''.join(map(str, etats[l])), 2) + 4**N, states[j].number()]
                    S_plus[i, j] += int(etats[l][mu + N*spin] == 0) * s1 * s2

            elif states[i].type == "r" and states[j].type == "k":    # cas <r|k>
                S_plus[i, j] = int(states[j].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]
            elif states[i].type == "k" and states[j].type == "r":    # cas <k|r>
                S_plus[i, j] = int(states[i].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]
            elif states[i].type == "k" and states[j].type == "k":    # cas <k|k>
                S_plus[i, j] = int(states[i].vector[mu + N*spin] == 0) * int(i == j)
    return S_plus

def H_plus_k(states, N, mu, spin, t, U, mu1):
    """
    Retourne la matrice H excitee en electrons
    pour une base de depart states, un systeme
    a N sites avec des parametres t, U et mu1
    et une excitation delocalisee en mu de spin 
    "spin"
    """  

    S = np.load(f"S_{N}.npy")

    # Creation de la matrice vide
    dim = len(states)
    H_plus = np.zeros((dim, dim), dtype=complex)

    # Creation de la matrice element par element
    for i in range(dim):
        for j in range(dim):
            if states[i].type == "r" and states[j].type == "r":    # cas <r|H|r>

                # Decomposition des etats r en etats k
                H_plus[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    for m in range(len(etats)):
                        delta = int(np.array_equal(etats[l], etats[m]))
                        term1 = U/N * sum(etats[m][N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                        term2 = HU_mu(etats[l], etats[m], N, mu, spin, U, "e")
                        term3 = -2*t * np.cos((2*np.pi*mu)/N) * delta
                        term4 = -mu1 * delta
                        term5 = ham_k(etats[l], etats[m], N, t, U, mu1)
                        s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2) + 4**N]
                        s2 = S[int(''.join(map(str, etats[m])), 2) + 4**N, states[j].number()]
                        H_plus[i, j] += s1 * s2 * (term2 + int(etats[m][mu + N*spin] == 0) * (term1 + term3 + term4 + term5)) 

            elif states[i].type == "r" and states[j].type == "k":    # cas <r|H|k>

                # Decomposition de l'etat r en etats k
                H_plus[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    delta = int(np.array_equal(etats[l], states[j].vector))
                    term1 = U/N * sum(states[j].vector[N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                    term2 = HU_mu(etats[l], states[j].vector, N, mu, spin, U, "e")
                    term3 = -2*t * np.cos((2*np.pi*mu)/N) * delta
                    term4 = -mu1 * delta
                    term5 = ham_k(etats[l], states[j].vector, N, t, U, mu1)
                    s = S[states[i].number(), int(''.join(map(str, etats[l])), 2) + 4**N]
                    H_plus[i, j] += s * (term2 + int(states[j].vector[mu + N*spin] == 0) * (term1 + term3 + term4 + term5))

            elif states[i].type == "k" and states[j].type == "r":    # cas <k|H|r>

                # Decomposition de l'etat r en etats k
                H_plus[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    delta = int(np.array_equal(states[i].vector, etats[l]))
                    term1 = U/N * sum(etats[l][N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                    term2 = HU_mu(states[i].vector, etats[l], N, mu, spin, U, "e")
                    term3 = -2*t * np.cos((2*np.pi*mu)/N) * delta
                    term4 = -mu1 * delta
                    term5 = ham_k(states[i].vector, etats[l], N, t, U, mu1)
                    s = S[int(''.join(map(str, etats[l])), 2) + 4**N, states[j].number()]
                    H_plus[i, j] += s * (term2 + int(etats[l][mu + N*spin] == 0) * (term1 + term3 + term4 + term5))

            elif states[i].type == "k" and states[j].type == "k":    # cas <k|H|k>
                delta = int(i == j)
                term1 = U/N * sum(states[j].vector[N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                term2 = HU_mu(states[i].vector, states[j].vector, N, mu, spin, U, "e")
                term3 = -2*t * np.cos((2*np.pi*mu)/N) * delta
                term4 = -mu1 * delta
                term5 = ham_k(states[i].vector, states[j].vector, N, t, U, mu1)
                H_plus[i, j] = term2 + int(states[j].vector[mu + N*spin] == 0) * (term1 + term3 + term4 + term5)

    return H_plus

def S_moins_k(states, N, mu, spin):
    """
    Retourne la metrique excitee en trous
    pour une base de depart states, un systeme
    a N sites et une excitation delocalisee 
    en mu de spin "spin"
    """

    S = np.load(f"S_{N}.npy")

    # Creation de la matrice vide
    dim = len(states)
    S_moins = np.zeros((dim, dim), dtype=complex)

    # Creation de la matrice element par element
    for i in range(dim):
        for j in range(dim):
            if states[i].type == "r" and states[j].type == "r":    # cas <r|r>

                # Decomposition de l'etat r en etats k
                S_moins[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2) + 4**N]
                    s2 = S[int(''.join(map(str, etats[l])), 2) + 4**N, states[j].number()]
                    S_moins[i, j] += int(etats[l][mu + N*spin] == 1) * s1 * s2

            elif states[i].type == "r" and states[j].type == "k":    # cas <r|k>
                S_moins[i, j] = int(states[j].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]
            elif states[i].type == "k" and states[j].type == "r":    # cas <k|r>
                S_moins[i, j] = int(states[i].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]
            elif states[i].type == "k" and states[j].type == "k":    # cas <k|k>
                S_moins[i, j] = int(states[i].vector[mu + N*spin] == 1) * int(i == j)
    return S_moins

def H_moins_k(states, N, mu, spin, t, U, mu1):
    """
    Retourne la matrice H excitee en trous
    pour une base de depart states, un systeme
    a N sites avec des parametres t, U et mu1
    et une excitation delocalisee en mu de spin 
    "spin"
    """

    S = np.load(f"S_{N}.npy")

    # Creation de la matrice vide
    dim = len(states)
    H_moins = np.zeros((dim, dim), dtype=complex)

    # Creation de la matrice element par element
    for i in range(dim):
        for j in range(dim):
            if states[i].type == "r" and states[j].type == "r":    # cas <r|H|r>

                # Decomposition des etats r en etats k
                H_moins[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    for m in range(len(etats)):
                        delta = int(np.array_equal(etats[l], etats[m]))
                        term1 = -U/N * sum(etats[m][N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                        term2 = HU_mu(etats[l], etats[m], N, mu, spin, U, "h")
                        term3 = 2*t * np.cos((2*np.pi*mu)/N) * delta
                        term4 = mu1 * delta
                        term5 = ham_k(etats[l], etats[m], N, t, U, mu1)
                        s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2) + 4**N]
                        s2 = S[int(''.join(map(str, etats[m])), 2) + 4**N, states[j].number()]
                        H_moins[i, j] += s1 * s2 * (term2 + int(etats[m][mu + N*spin] == 1) * (term1 + term3 + term4 + term5)) 
                     
            elif states[i].type == "r" and states[j].type == "k":    # cas <r|H|k>

                # Decomposition de l'etat r en etats k
                H_moins[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    delta = int(np.array_equal(etats[l], states[j].vector))
                    term1 = -U/N * sum(states[j].vector[N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                    term2 = HU_mu(etats[l], states[j].vector, N, mu, spin, U, "h")
                    term3 = 2*t * np.cos((2*np.pi*mu)/N) * delta
                    term4 = mu1 * delta
                    term5 = ham_k(etats[l], states[j].vector, N, t, U, mu1)
                    s = S[states[i].number(), int(''.join(map(str, etats[l])), 2) + 4**N]
                    H_moins[i, j] += s * (term2 + int(states[j].vector[mu + N*spin] == 1) * (term1 + term3 + term4 + term5))

            elif states[i].type == "k" and states[j].type == "r":    # cas <k|H|r>

                # Decomposition de l'etat r en etats k
                H_moins[i, j] = 0
                etats = etats_decomposes(N)
                for l in range(len(etats)):
                    delta = int(np.array_equal(states[i].vector, etats[l]))
                    term1 = -U/N * sum(etats[l][N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                    term2 = HU_mu(states[i].vector, etats[l], N, mu, spin, U, "h")
                    term3 = 2*t * np.cos((2*np.pi*mu)/N) * delta
                    term4 = mu1 * delta
                    term5 = ham_k(states[i].vector, etats[l], N, t, U, mu1)
                    s = S[int(''.join(map(str, etats[l])), 2) + 4**N, states[j].number()]
                    H_moins[i, j] += s * (term2 + int(etats[l][mu + N*spin] == 1) * (term1 + term3 + term4 + term5))

            elif states[i].type == "k" and states[j].type == "k":    # cas <k|H|k>
                delta = int(i == j)
                term1 = -U/N * sum(states[j].vector[N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                term2 = HU_mu(states[i].vector, states[j].vector, N, mu, spin, U, "h")
                term3 = 2*t * np.cos((2*np.pi*mu)/N) * delta
                term4 = mu1 * delta
                term5 = ham_k(states[i].vector, states[j].vector, N, t, U, mu1)
                H_moins[i, j] = term2 + int(states[j].vector[mu + N*spin] == 1) * (term1 + term3 + term4 + term5)
    return H_moins

def mix_green(r, k, N, spin, t, U, mu, basis):
    """
    Prend en entree les etats de la base mixte
    et retourne la densite d'etats pour un 
    systeme de N sites avec des parametres
    t, U et mu excite par des electrons/trous
    de la base "basis"
    """
    
    # Creation des etats de la base mixte
    states = []
    for etat_r in r:
        ket_r = Ket(list(map(int, format(etat_r, f'0{2*N}b'))), "r", str(etat_r) + "r")
        states.append(ket_r)
    for etat_k in k:
        ket_k = Ket(list(map(int, format(etat_k, f'0{2*N}b'))), "k", str(etat_k) + "k")
        states.append(ket_k)

    # Diagonalisation dans le bloc de depart
    E, H, S, omega, overfilled = ground_energy(states, N, t, U, mu)
    E0 = E[0]    # energie fondamentale
    Omega = omega[0]    # etat fondamental

    # Creation des elements diagonaux de la matrice de Green
    w = np.linspace(-10, 10, 2000)
    dos = np.zeros(len(w))

    # Fonction de Green G_ii
    for i in range(N):
        
        # Matrices dans le sous-espace excite en electrons
        if basis == "r":
            S_plus = S_plus_r(states, N, i, spin)
            H_plus = H_plus_r(states, N, i, spin, t, U, mu)
        elif basis == "k":
            S_plus = S_plus_k(states, N, i, spin)
            H_plus = H_plus_k(states, N, i, spin, t, U, mu)

        # Verification de la surcompletude de la base (valeurs propres de S)
        eig_S_plus = np.linalg.eigh(S_plus)[0]

        # Au moins une valeur propre de S nulle => diagonalisation generalisee
        if np.any(np.isclose(eig_S_plus, 0, atol=1e-6)):  
            e_eigvals, e_eigenvectors = gen_diagonalization(H_plus, S_plus)       

        # Aucune valeur propre de S nulle => diagonalisation par eigh
        else:
            e_eigvals, e_eigenvectors = sp.linalg.eigh(H_plus, S_plus)

        # Matrices dans le sous-espace excite en trous       
        if basis == "r":
            S_moins = S_moins_r(states, N, i, spin)
            H_moins = H_moins_r(states, N, i, spin, t, U, mu)
        if basis == "k":
            S_moins = S_moins_k(states, N, i, spin)
            H_moins = H_moins_k(states, N, i, spin, t, U, mu)

        # Verification de la surcompletude de la base (valeurs propres de S)
        eig_S_moins = np.linalg.eigh(S_moins)[0]

        # Au moins une valeur propre de S nulle => diagonalisation generalisee
        if np.any(np.isclose(eig_S_moins, 0, atol=1e-6)):
            h_eigvals, h_eigenvectors = gen_diagonalization(H_moins, S_moins)       

        # Aucune valeur propre de S nulle => diagonalisation par eigh
        else:
            h_eigvals, h_eigenvectors = sp.linalg.eigh(H_moins, S_moins)

        # Creation des vecteurs de poids Q
        Q_e = Omega.conj().T @ S_plus @ e_eigenvectors
        Q_h = Omega.conj().T @ S_moins @ h_eigenvectors

        # Calcul de la partie imaginaire de la fonction de Green
        g = green(w, Q_e, Q_h, e_eigvals, h_eigvals, E0, N)       
        dos += g

    #plt.plot(w, g + i, "r")
    #plt.show()

    return w, dos

def exact_green_2sites(t, U, mu):
    """
    Exemple analytique qui retourne la 
    densite d'etats exacte pour un systeme
    a 2 sites avec des parametres t, U et mu
    """
    
    # Creation des etats de la base de depart
    etats = [5, 6, 9, 10]    
    states = []
    for etat_r in etats:
        ket_r = Ket(list(map(int, format(etat_r, f'0{2*N}b'))), "r", str(etat_r) + "r")
        states.append(ket_r)

    # Diagonalisation dans le bloc de depart
    E, H, S, omega, overfilled = ground_energy(states, 2, t, U, mu)
    E0 = E[0]    # energie fondamentale
    Omega = omega[0]    # etat fondamental

    # Creation des elements diagonaux de la matrice de Green
    w = np.linspace(-10, 10, 2000)
    dos = np.zeros(len(w))

    # Creation des matrices H excitees vides
    H_plus = np.zeros((2, 2), dtype=complex)
    H_moins = np.zeros((2, 2), dtype=complex)

    # Creation des etats des sous-espaces excites up
    etats_plus = [13, 14]
    etats_moins = [1, 2]
    states_plus = []
    states_moins = []
    for etat_r in etats_plus:
        ket_r = Ket(list(map(int, format(etat_r, f'0{2*N}b'))), "r", str(etat_r) + "r")
        states_plus.append(ket_r)
    for etat_r in etats_moins:
        ket_r = Ket(list(map(int, format(etat_r, f'0{2*N}b'))), "r", str(etat_r) + "r")
        states_moins.append(ket_r)

    # Creation des matrices excitees element par element
    for i in range(2):
        for j in range(2):
            H_plus[i, j] = ham_r(states_plus[i].vector, states_plus[j].vector, 2, t, U, mu)
            H_moins[i, j] = ham_r(states_moins[i].vector, states_moins[j].vector, 2, t, U, mu)

    # Diagonalisation dans les blocs excites
    e_eigvals, e_eigvecs = np.linalg.eigh(H_plus)
    h_eigvals, h_eigvecs = np.linalg.eigh(H_moins)

    # Fonction de Green G_ii
    for i in range(2):
        
        # Etat fondamental dans les bases excitees
        if i == 0:
            omega_plus = Omega[:2]
            omega_moins = Omega[2:]
        elif i == 1:
            omega_plus = Omega[2:]
            omega_moins = Omega[:2]

        # Creation des vecteurs de poids Q
        Q_e = omega_plus.conj().T @ e_eigvecs
        Q_h = omega_moins.conj().T @ h_eigvecs

        # Calcul de la partie imaginaire de la fonction de Green
        g = green(w, Q_e, Q_h, e_eigvals, h_eigvals, E0, 2)        
        dos += g

    return w, dos

def green(x, Q_e, Q_h, e_eigs, h_eigs, E0, N, eta=0.05j):
    """
    Prend en entree une variable independante x, 
    les matrices de poids Q_e/Q_h, les valeurs
    propres des sous-espaces excites e_eigs/h_eigs,
    l'energie fondamentale E0 et la taille du
    systeme N et retourne la partie imaginaire
    (ponderee par un facteur -1/Npi) de la 
    fonction de Green
    """
    
    green_function = np.zeros(len(x))

    # Calcul de la valeur de green_function pour chaque x
    for i in range(len(x)):
        value = 0
        for j in range(len(Q_e)):    # partie excitee en electrons
            value += np.abs(Q_e[j])**2/(x[i] + eta + E0 - e_eigs[j]) 
        for j in range(len(Q_h)):    # partie excitee en trous
            value += np.abs(Q_h[j])**2/(x[i] + eta - E0 + h_eigs[j])   
        green_function[i] = value.imag/(-N*np.pi)
    
    return green_function

## Graphique de la DOS pour les parametres donnes (en haut)

#w, dos = mix_green(r, k, N, spin, t, U, mu, basis)
#plt.plot(w, dos, "r")
#plt.xlabel(r"$\omega$")
#plt.ylabel(r"$n(\omega)$")
#plt.title(rf"#sites = {N}    $U = {U}$    $\mu = {mu}$    $N = {N}$    $S_z = 0$")
#plt.grid()
#plt.show()

## Graphique des DOS (r, k et exact) (ne fonctionne que pour N = 4, t = 1 et U = 2, 4, 8, ou 12)

#w_k, dos_k = mix_green(r, k, N, spin, t, U, mu, "k")
#w_r, dos_r = mix_green(r, k, N, spin, t, U, mu, "r")
#data = np.loadtxt(f"dos_u{U}", skiprows=2)
#x = data[:, 0]
#y1 = data[:, 1]
#y2 = data[:, 2]
#plt.plot(x, y1 + y2, "black", label="exact")
#plt.plot(w_r, dos_r, "r", label="r")
#plt.plot(w_k, dos_k, "b", label="k")
#plt.xlabel(r'$\omega$')
#plt.ylabel(r'$n(\omega)$')
#plt.title(rf"#sites = {N}    $U = {U}$    $\mu = {mu}$    $N = {N}$    $S_z = 0$")
#plt.legend()
#plt.grid()
#plt.show()

## Graphique de 4 DOS (r, k et exact) (U = 2, 4, 8, 12) en une figure

#U_values = [2, 4, 8, 12]
#fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
#for ax, U in zip(axes, U_values):
#    w_k, dos_k = mix_green(r, k, N, spin, t, U, U/2, "k")
#    w_r, dos_r = mix_green(r, k, N, spin, t, U, U/2, "r")
#    data = np.loadtxt(f"dos_u{U}", skiprows=2)
#    x = data[:, 0]
#    y1 = data[:, 1]
#    y2 = data[:, 2]
#    ax.plot(x, y1 + y2, "black", label="exact")
#    ax.plot(w_r, dos_r, "r", label="r")
#    ax.plot(w_k, dos_k, "b", label="k")
#    ax.set_ylabel(r"$n(\omega)$")
#    ax.set_title(rf"$U = {U}$", loc="left")
#    ax.legend()
#    ax.grid()
#axes[-1].set_xlabel(r"$\omega$")
#fig.suptitle(rf"#sites = {N}    $t = {t}$    $\mu = U / 2$    $S_z = 0$", fontsize=14)
#plt.tight_layout()
#plt.show()
