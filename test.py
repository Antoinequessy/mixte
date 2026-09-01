import os
import math
import numpy as np
import scipy as sp
from sympy import symbols, prod
import matplotlib.pyplot as plt
import time
from itertools import combinations, product
from collections import deque

np.set_printoptions(precision=5, linewidth=150, suppress=True)


class Ket:
    def __init__(self, vector, typ, name=None, compteur=0):
        self.type = typ
        self.vector = np.array(vector)
        self.name = name
        self.compteur = compteur

    def number(self):
        return int("".join(map(str, self.vector)), 2) + int(self.type == "k") * 4**(len(self.vector)//2)

    def __repr__(self):
        return self.name if self.name is not None else "Ket"



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



def m_get_H_S(N, t=1, U=2):

    H_file = f"m_H_{N}_{t}_{U}.npy"
    S_file = f"m_S_{N}.npy"

    if os.path.exists(H_file) and os.path.exists(S_file):

        H = np.load(H_file)
        S = np.load(S_file)
        
        return H, S

    etats = np.arange(4**(2*N))

    states = []

    compteur = 0

    for element in etats:

        lst = list(map(int, format(element, f'0{4*N}b')))

        if sum(lst[:2*N]) != N/2 or sum(lst[2*N:]) != N/2:
            continue

        ket = Ket(lst, None, str(element), compteur=compteur)
        states.append(ket)

        compteur += 1

    _, H, S, _, _ = m_ground_energy(states, N, t, U)

    np.save(H_file, H)
    np.save(S_file, S)

    return H, S



def get_H_S(N, t=1, U=2):

    H_file = f"H_{N}_{t}_{U}.npy"
    S_file = f"S_{N}.npy"

    if os.path.exists(H_file) and os.path.exists(S_file):

        H = np.load(H_file)
        S = np.load(S_file)
        
        return H, S

    H, S = complete_matrixes(N, t, U)

    np.save(H_file, H)
    np.save(S_file, S)

    return H, S



def gen_diagonalization(H, S, eps=1e-6):

    eig_S, U = np.linalg.eigh(S)

    mask = eig_S > eps

    eig_S = eig_S[mask]
    U = U[:, mask]

    W = U @ np.diag(1/np.sqrt(eig_S))

    H_red = W.conj().T @ H @ W

    eigenvalues, Y = np.linalg.eigh(H_red)

    eigenvectors = W @ Y

    return eigenvalues, eigenvectors



def split_state(state, N):

    r_up, k_up, r_down, k_down = (state[i*N:(i+1)*N] for i in range(4))

    return r_up, k_up, r_down, k_down



def replace_zero_combinations(lst, r, N):
    lst = np.array(lst)
    n = sum(r)
    
    zero_pos = np.where(lst == 0)[0]
    one_pos  = np.where(lst == 1)[0]

    k_states = []
    sigma   = []
    combs = []

    for c in combinations(zero_pos, n):

        compteur = np.sum(one_pos < np.array(c)[:, None])

        sigma.append(compteur)

        new_lst = lst.copy()
        new_lst[list(c)] = 1
        k_states.append(new_lst)

        combs.append(c)

    r_pos = np.flatnonzero(r)
    factor = 1j * (2*np.pi/N)

    c = []

    for element, s in zip(combs, sigma):

        k_pos = element
        mat = np.exp(factor * np.outer(r_pos, k_pos))
        c.append((-1)**s * np.linalg.det(mat))

    return k_states, sigma, c



def met_m(state1, state2, N):

    r_up1, k_up1, r_down1, k_down1 = split_state(state1, N)
    r_up2, k_up2, r_down2, k_down2 = split_state(state2, N)

    n_up1 = sum(r_up1) + sum(k_up1)
    n_down1 = sum(r_down1) + sum(k_down1)
    n_up2 = sum(r_up2) + sum(k_up2)
    n_down2 = sum(r_down2) + sum(k_down2)

    if (n_up1 > N or n_up2 > N or n_down1 > N or n_down2 > N
        or n_up1 != n_up2 or n_down1 != n_down2):

        err = True

        return 0, [], [], [], [], 0, err

    k_up_states1, sig_up1, c_up1       = replace_zero_combinations(k_up1, r_up1, N)
    k_down_states1, sig_down1, c_down1 = replace_zero_combinations(k_down1, r_down1, N)
    k_up_states2, sig_up2, c_up2       = replace_zero_combinations(k_up2, r_up2, N)
    k_down_states2, sig_down2, c_down2 = replace_zero_combinations(k_down2, r_down2, N)

    k_states1 = [list(a) + list(b) for a, b in product(k_up_states1, k_down_states1)]
    k_states2 = [list(a) + list(b) for a, b in product(k_up_states2, k_down_states2)]

    c1 = [a*b for a, b in product(c_up1, c_down1)]
    c2 = [a*b for a, b in product(c_up2, c_down2)]

    d2 = {}
    for j, b in enumerate(k_states2):
        key = tuple(b)
        d2.setdefault(key, []).append(j)

    matches = [(i, j) for i, a in enumerate(k_states1) for j in d2.get(tuple(a), [])]

    somme = sum(np.conj(c1[i]) * c2[j] for i, j in matches)

    norm = N**(-0.5 * (sum(r_up1) + sum(r_down1) + sum(r_up2) + sum(r_down2)))

    err = False

    return norm * somme, k_states1, c1, k_states2, c2, norm, err


def m_ground_energy(states, N, t=1, U=2, mu=0):

    dim = len(states)

    H = np.zeros((dim, dim), dtype=complex)
    S = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(i, dim):
             
            S[i, j], k_states1, c1, k_states2, c2, norm, err = met_m(states[i].vector, states[j].vector, N)

            if err == True:
                
                H[i, j] = 0
                
                continue

            H[i, j] = 0

            for l in range(len(k_states1)):
                for m in range(len(k_states2)):

                    H[i, j] += np.conj(c1[l]) * c2[m] * ham_k(k_states1[l], k_states2[m], N, t, U, mu)

            H[i, j] = norm * H[i, j]
    
    new_H = H + H.conj().T
    np.fill_diagonal(new_H, np.diag(H).real)

    new_S = S + S.conj().T
    np.fill_diagonal(new_S, np.diag(S).real)

    eig_S = np.linalg.eigh(new_S)[0]

    if np.any(np.isclose(eig_S, 0, atol=1e-6)):
        
        eigenvalues, eigenvectors = gen_diagonalization(new_H, new_S)        
        overfilled = True

    else:

        eigenvalues, eigenvectors = sp.linalg.eigh(new_H, new_S)
        overfilled = False

    return eigenvalues, new_H, new_S, eigenvectors.T, overfilled



def m_optimized_ground_energy(states, H, S, N, t=1, U=2):

    n = []
    
    for element in states:
        n.append(int(element.compteur))

    new_H = H[np.ix_(n, n)]
    new_S = S[np.ix_(n, n)]

    eig_S = np.linalg.eigh(new_S)[0]

    if np.any(np.isclose(eig_S, 0, atol=1e-6)):
        
        eigenvalues, eigenvectors = gen_diagonalization(new_H, new_S)        
        overfilled = True

    else:

        eigenvalues, eigenvectors = sp.linalg.eigh(new_H, new_S)
        overfilled = False

    return eigenvalues, new_H, new_S, eigenvectors.T, overfilled



def metrique_r(N):
    return np.identity(4**N, dtype=int)



def metrique_k(N):
    return np.identity(4**N, dtype=int)



def metrique_rk(N):

    dim = 4**N
    matrice = np.zeros((dim, dim), dtype=complex)

    occ = ((np.arange(dim)[:, None] >> np.arange(2*N)) & 1)[:, ::-1]

    indices = np.arange(2*N)
    F = np.exp(2j * np.pi * np.outer(indices, indices) / N)

    for r in range(dim):
        occ_r = occ[r]
        n1_r = np.where(occ_r[:N] == 1)[0]
        n2_r = np.where(occ_r[N:] == 1)[0] + N

        for k in range(dim):
            occ_k = occ[k]
            n1_k = np.where(occ_k[:N] == 1)[0]
            n2_k = np.where(occ_k[N:] == 1)[0] + N

            if len(n1_r) != len(n1_k) or len(n2_r) != len(n2_k):
                continue

            mat1 = F[np.ix_(n1_r, n1_k)]
            mat2 = F[np.ix_(n2_r, n2_k)]

            power = 0.5 * (len(n1_r) + len(n2_r))
            el = (1 / N**power) * np.linalg.det(mat1) * np.linalg.det(mat2)

            matrice[r, k] = np.round(el.real, 5) + 1j*np.round(el.imag, 5)

    return matrice



def hamiltonian_r(N, t=1, U=2):

    dim = 4 ** N

    H = np.zeros((dim, dim), dtype=float)

    etats = [list(map(int, format(i, f'0{2*N}b'))) for i in range(dim)]

    index = {tuple(e): k for k, e in enumerate(etats)}
    
    for element in etats:

        i0 = index[tuple(element)]

        element_up = element[:N]
        element_down = element[N:]

        n_up = sum (element_up)
        n_down = sum(element_down)

        n_U = 0

        for i in range(N):

            if element[i] == 1 and element[i + N] == 1:

                n_U += 1
                
            if element_up[i] == 1 and element_up[i - 1] == 0:

                new_state = element.copy()

                new_state[i] = 0

                if i != 0:
                    new_state[i - 1] = 1

                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t

                elif i == 0:
                    new_state[N - 1] = 1

                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t * (-1) ** (n_up - 1)


            if element_down[i] == 1 and element_down[i - 1] == 0:

                new_state = element.copy()

                new_state[i + N] = 0
                
                if i != 0:
                    new_state[i + N - 1] = 1

                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t

                elif i == 0:
                    new_state[-1] = 1

                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t * (-1) ** (n_down - 1)


            if element_up[i] == 0 and element_up[i - 1] == 1:

                new_state = element.copy()

                new_state[i] = 1

                if i != 0:
                    new_state[i - 1] = 0

                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t

                elif i == 0:
                    new_state[N - 1] = 0

                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t * (-1) ** (n_up - 1)
                    

            if element_down[i] == 0 and element_down[i - 1] == 1:

                new_state = element.copy()

                new_state[i + N] = 1
                
                if i != 0:
                    new_state[i + N - 1] = 0

                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t

                elif i == 0:
                    new_state[-1] = 0

                    j0 = index[tuple(new_state)]
                    H[i0, j0] = -t * (-1) ** (n_down - 1)
                    

        H[i0, i0] = n_U * U        
   
    return H



def hamiltonian_k(N, t=1, U=2):

    dim = 4 ** N

    H = np.zeros((dim, dim), dtype=float)

    etats = [list(map(int, format(i, f'0{2*N}b'))) for i in range(dim)]

    index = {tuple(e): k for k, e in enumerate(etats)}
    
    for element in etats:

        i0 = index[tuple(element)]

        element_up = element[:N]
        element_down = element[N:]

        n_up = sum(element_up)
        n_down = sum(element_down)

        somme = 0
        
        for i in range(N):

            somme += np.cos((2*np.pi*i) / N) * (element_up[i] + element_down[i])

        H[i0, i0] = -2*t * somme + U/N * n_up * n_down

        for i in range(N):

            if element_up[i] == 1:
                
                for j in range(N):

                    if element_up[j] == 0:

                        if j > i:
                            q = i - j + N

                            zwizz = 1
                            
                        elif j < i:
                            q = i - j

                            zwizz = 0
                            
                        for k in range(N):

                            if element_down[k] == 0 and element_down[k - q] == 1:

                                    new_state = element.copy()

                                    new_state[i] = 0
                                    new_state[j] = 1
                                    new_state[k + N] = 1

                                    if k - q >= 0:
                                        new_state[k - q + N] = 0

                                        j0 = index[tuple(new_state)]

                                        H[i0, j0] = U/N * (-1) ** (sum(element_up[:i]) + sum(element_up[:j]) + sum(element_down[:k]) + sum(element_down[:(k-q)]) - zwizz - 1)
                                        
                                        
                                    elif k - q < 0:
                                        new_state[k - q] = 0

                                        j0 = index[tuple(new_state)]

                                        H[i0, j0] = U/N * (-1) ** (sum(element_up[:i]) + sum(element_up[:j]) + sum(element_down[:k]) + sum(element_down[:(N+k-q)]) - zwizz)
                                        
    return H



def hamiltonian_rk(N, t=1, U=2):

    dim = 4 ** N

    H_t = np.zeros(dim, dtype=float)
    H_U = np.zeros(dim, dtype=float)

    etats = [list(map(int, format(i, f'0{2*N}b'))) for i in range(dim)]

    index = {tuple(e): k for k, e in enumerate(etats)}


    for i in range(dim):     
        for j in range(N):
        
            H_t[i] += -2*t * np.cos((2*np.pi*j)/N) * (etats[i][j] + etats[i][j+N])
            H_U[i] += U * etats[i][j] * etats[i][j+N]

    return np.add.outer(H_t, H_U)



def complete_matrixes(N, t=1, U=2):

    S_r = metrique_r(N)
    S_k = metrique_k(N)
    S_rk = metrique_rk(N)
    S_kr = S_rk.conj().T

    S = np.block([[S_r,  S_kr],
                  [S_rk, S_k]])

    H_r = hamiltonian_r(N, t, U)
    H_k = hamiltonian_k(N, t, U)
    H_rk = np.multiply(hamiltonian_rk(N, t, U), metrique_rk(N))
    H_kr = H_rk.conj().T

    H = np.block([[H_r,  H_kr],
                  [H_rk, H_k]])

    return H, S



def ham_k(state1, state2, N, t=1, U=2, mu=0):

    if np.array_equal(state1, state2):

        state_up = state1[:N]
        state_down = state1[N:]
    
        n_up = sum(state_up)
        n_down = sum(state_down)

        somme = 0
        
        for i in range(N):

            somme += np.cos((2*np.pi*i) / N) * (state_up[i] + state_down[i])
        
        H = -2*t * somme + U/N * n_up * n_down - mu * sum(state1)

        return H

    state1_up = state1[:N]
    state1_down = state1[N:]
    
    n_up1 = sum(state1_up)
    n_down1 = sum(state1_down)

    state2_up = state2[:N]
    state2_down = state2[N:]
    
    n_up2 = sum(state2_up)
    n_down2 = sum(state2_down)

    if n_up1 != n_up2 or n_down1 != n_down2:

        H = 0

        return H
        
    k_up = []
    k_up_dag = []
    k_down = []
    k_down_dag = []

    for i in range(N):

        if state1_up[i] == 0 and state2_up[i] == 1:

            k_up_dag.append(i)

        elif state1_up[i] == 1 and state2_up[i] == 0:

            k_up.append(i)

        if state1_down[i] == 0 and state2_down[i] == 1:

            k_down_dag.append(i)

        elif state1_down[i] == 1 and state2_down[i] == 0:

            k_down.append(i)

    if len(k_up_dag) != 1 or len(k_down_dag) != 1 or len(k_up) != 1 or len(k_down) != 1:

        H = 0

        return H

    if k_up[0] > k_up_dag[0]:

        q_up = k_up[0] - k_up_dag[0]

        zwizz_up = 0

    elif k_up[0] < k_up_dag[0]:

        q_up = k_up[0] - k_up_dag[0] + N

        zwizz_up = 1

    if k_down[0] > k_down_dag[0]:

        q_down = k_down_dag[0] - k_down[0] + N

        zwizz_down = 0

    elif k_down[0] < k_down_dag[0]:

        q_down = k_down_dag[0] - k_down[0]

        zwizz_down = 1

    if q_up != q_down:

        H = 0

        return H

    H = U/N * (-1) ** (sum(state1_up[:k_up[0]]) + sum(state1_up[:k_up_dag[0]]) + sum(state1_down[:k_down[0]]) + sum(state1_down[:k_down_dag[0]]) - zwizz_up - zwizz_down)

    return H



def ham_r(state1, state2, N, t=1, U=2, mu=0):

    if np.array_equal(state1, state2):

        state_up = state1[:N]
        state_down = state1[N:]

        n_U = 0

        for i in range(N):
                       
            if state_up[i] == 1 and state_down[i] == 1:

                n_U += 1

        H = U * n_U - mu * (sum(state1))

        return H

    state1_up = state1[:N]
    state1_down = state1[N:]

    n_up1 = sum(state1_up)
    n_down1 = sum(state1_down)

    state2_up = state2[:N]
    state2_down = state2[N:]
    
    n_up2 = sum(state2_up)
    n_down2 = sum(state2_down)

    if n_up1 != n_up2 or n_down1 != n_down2:

        H = 0

        return H

    index_up = []
    index_down = []

    for i in range(N):

        if state1_up[i] != state2_up[i]:

            index_up.append(i)

        if state1_down[i] != state2_down[i]:

            index_down.append(i)

    if not ((len(index_up) == 2 and len(index_down) == 0) or (len(index_up) == 0 and len(index_down) == 2)):

        H = 0

        return H

    if len(index_up) == 2:

        if index_up[1] - index_up[0] == N - 1:

             zwizz = n_up1 - 1

        elif index_up[1] - index_up[0] == 1:

            zwizz = 0

        else:

            H = 0

            return H

    elif len(index_down) == 2:

        if index_down[1] - index_down[0] == N - 1:

             zwizz = n_down1 - 1

        elif index_down[1] - index_down[0] == 1:

            zwizz = 0

        else:

            H = 0

            return H


    H = -t * (-1) ** (zwizz)

    return H



def ham_rk(order, state1, state2, N, t=1, U=2, mu=0):

    state1_up = state1[:N]
    state1_down = state1[N:]
    
    n_up1 = sum(state1_up)
    n_down1 = sum(state1_down)

    state2_up = state2[:N]
    state2_down = state2[N:]
    
    n_up2 = sum(state2_up)
    n_down2 = sum(state2_down)

    if n_up1 != n_up2 or n_down1 != n_down2:

        H = 0

        return H
    
    H_t = 0

    for i in range(N):
        
        if order == "rk":
        
            H_t += -2*t * np.cos((2*np.pi*i)/N) * (state2[i] + state2[i + N])

        elif order == "kr":
        
            H_t += -2*t * np.cos((2*np.pi*i)/N) * (state1[i] + state1[i + N])

    H_U = 0

    for i in range(N):

        if order == "rk":

            H_U += U * state1[i] * state1[i + N]

        if order == "kr":

            H_U += U * state2[i] * state2[i + N]

    H_mu = - mu * sum(state1)
            

    H = met_rk(order, state1, state2, N) * (H_t + H_U + H_mu)
    
    return H



def met_r(state1, state2, N):

    if np.array_equal(state1, state2):

        S = 1

        return S

    S = 0

    return S


def met_k(state1, state2, N):

    if np.array_equal(state1, state2):

        S = 1

        return S

    S = 0

    return S


def met_rk(order, state1, state2, N):

    state1_up   = state1[:N]
    state1_down = state1[N:]

    state2_up   = state2[:N]
    state2_down = state2[N:]

    n_up   = np.count_nonzero(state1_up)
    n_down = np.count_nonzero(state1_down)

    if (n_up != np.count_nonzero(state2_up) or
        n_down != np.count_nonzero(state2_down)):
        return 0.0

    if order == "rk":

        r_up   = np.flatnonzero(state1_up)
        k_up   = np.flatnonzero(state2_up)

        r_down = np.flatnonzero(state1_down)
        k_down = np.flatnonzero(state2_down)

        factor = -1j * (2*np.pi/N)

    elif order == "kr":

        k_up   = np.flatnonzero(state1_up)
        r_up   = np.flatnonzero(state2_up)

        k_down = np.flatnonzero(state1_down)
        r_down = np.flatnonzero(state2_down)

        factor = 1j * (2*np.pi/N)

    det_up = np.linalg.det(np.exp(factor * np.outer(r_up, k_up)))
    det_down = np.linalg.det(np.exp(factor * np.outer(r_down, k_down)))

    return N**(-(n_up+n_down)/2) * det_up * det_down



def ground_energy(states, N, t=1, U=2, mu=0):   
    
    dim = len(states)

    H = np.zeros((dim, dim), dtype=complex)
    S = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(i, dim):

            if states[i].type == "r" and states[j].type == "r":
                
                H[i, j] = ham_r(states[i].vector, states[j].vector, N, t, U, mu)
                S[i, j] = met_r(states[i].vector, states[j].vector, N)

            elif states[i].type == "r" and states[j].type == "k":
                
                H[i, j] = ham_rk("rk", states[i].vector, states[j].vector, N, t, U, mu)
                S[i, j] = met_rk("rk", states[i].vector, states[j].vector, N)

            elif states[i].type == "k" and states[j].type == "r":
                
                H[i, j] = ham_rk("kr", states[i].vector, states[j].vector, N, t, U, mu)
                S[i, j] = met_rk("kr", states[i].vector, states[j].vector, N)

            elif states[i].type == "k" and states[j].type == "k":
                
                H[i, j] = ham_k(states[i].vector, states[j].vector, N, t, U, mu)
                S[i, j] = met_k(states[i].vector, states[j].vector, N)
    
    new_H = H + H.conj().T
    np.fill_diagonal(new_H, np.diag(H).real)

    new_S = S + S.conj().T
    np.fill_diagonal(new_S, np.diag(S).real)

    eig_S = np.linalg.eigh(new_S)[0]

    if np.any(np.isclose(eig_S, 0, atol=1e-6)):
        
        eigenvalues, eigenvectors = gen_diagonalization(new_H, new_S)        
        overfilled = True

    else:

        eigenvalues, eigenvectors = sp.linalg.eigh(new_H, new_S)
        overfilled = False

    return eigenvalues, new_H, new_S, eigenvectors.T, overfilled




def optimized_ground_energy(states, H, S, N, t=1, U=2):

    n = []
    
    for element in states:
        if element.type == "r":
            n.append(element.number())
        if element.type == "k":
            n.append(element.number())

    new_H = H[np.ix_(n, n)]
    new_S = S[np.ix_(n, n)]

    eig_S = np.linalg.eigh(new_S)[0]

    if np.any(np.isclose(eig_S, 0, atol=1e-4)):
        
        eigenvalues, eigenvectors = gen_diagonalization(new_H, new_S)        
        overfilled = True

    else:

        eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(new_H, 1, new_S, which="SA")
        overfilled = False

    return eigenvalues, new_H, new_S, eigenvectors.T, overfilled



def small_optimized_ground_energy(states, H, S, N, t=1, U=2):

    n = []
    
    for element in states:
        if element.type == "r":
            n.append(element.number())
        if element.type == "k":
            n.append(element.number())

    new_H = H[np.ix_(n, n)]
    new_S = S[np.ix_(n, n)]

    eig_S = np.linalg.eigh(new_S)[0]

    if np.any(np.isclose(eig_S, 0, atol=1e-4)):
        
        eigenvalues, eigenvectors = gen_diagonalization(new_H, new_S)        
        overfilled = True

    else:

        eigenvalues, eigenvectors = sp.linalg.eigh(new_H, new_S)
        overfilled = False

    return eigenvalues, new_H, new_S, eigenvectors.T, overfilled



def complete_basis(S, states, N):
    """
    Prend en entree une matrice complete S et
    l'ensemble des etats d'un bloc a demi-rempli
    en r et en k.
    Retourne une base mixte aleatoire qui n'est pas
    surcomplete.
    """

    n = []
    
    for element in states:
        if element.type == "r":
            n.append(element.number())
        if element.type == "k":
            n.append(element.number())

    n = np.array(n)

    eigvals = [0]
    
    while np.isclose(eigvals[0], 0, atol=1e-6):

        comb = np.sort(np.random.choice(len(states), size=len(states)//2, replace=False))

        new_S = S[np.ix_(n[comb], n[comb])]
        eigvals, _ = np.linalg.eigh(new_S)

    return [states[i] for i in comb]





## Test 2-sites : 3 etats


etats = [5, 6, 9, 10]

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


def test_2(t=1, U=2):

    for etats in combinations(states, 3):
        
        E0 = ground_energy(list(etats), 2, t, U)[0]

        if E0 < -3.12:

            print([s.name for s in etats])
            print(round(E0, 5))


#test_2() 




## Test 4-sites : sous-espaces r/k et energies fondamentales associees


#H_r = hamiltonian_r(4)
#
#r_states = np.arange(256)
#etats_r = []
#
#for r in r_states:
#    
#    ket_r = Ket(list(map(int, format(r, f'0{8}b'))), "r", str(r) + "r")
#    etats_r.append(ket_r)
#
#H_r = np.zeros((256, 256), dtype=complex)
#
#for i in range(256):
#    for j in range(256):
#
#        H_r[i, j] = ham_r(etats_r[i].vector, etats_r[j].vector, 4, 1, 2, 1)
#
#comps_r = all_components(H_r)
#
#fund_r = []
#
#for c in comps_r:
#
#    print(c)
#
#    S_sorted = sorted(c)
#    H_sub = H_r[np.ix_(S_sorted, S_sorted)]
#
#    eigvals = []
#
#    for val in np.linalg.eigvals(H_sub): 
#        eigvals.append(float(round(val.real, 5)))
#
#    for el in eigvals:
#        fund_r.append(el)
#
#    print(sorted(eigvals))
#
#print(sorted(fund_r))
#
#    
#H_k = hamiltonian_k(4)
#
#k_states = np.arange(256)
#etats_k = []
#
#for k in k_states:
#    
#    ket_k = Ket(list(map(int, format(k, f'0{8}b'))), "k", str(k) + "k")
#    etats_k.append(ket_k)
#
#H_k = np.zeros((256, 256), dtype=complex)
#
#for i in range(256):
#    for j in range(256):
#
#        H_k[i, j] = ham_k(etats_k[i].vector, etats_k[j].vector, 4, 1, 4, 2)
#
#
#comps_k = all_components(H_k)
#
#fund_k = []
#
#for c in comps_k:
#
#    print(c)
#
#    S_sorted = sorted(c)
#    H_sub = H_k[np.ix_(S_sorted, S_sorted)]
#
#    eigvals = []
#
#    for val in np.linalg.eigvals(H_sub): 
#        eigvals.append(float(round(val.real, 5)))
#
#    for el in eigvals:
#        fund_k.append(el)
#
#    print(sorted((eigvals)))
#
#print(sorted(fund_k))
#



## Test 4-sites : base mixte de 2 a 8 etats (tests si mieux que k pur)


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

ensemble_minimal = []

states = []

for element in etats1:
    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")

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
    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")

    states_r2.append(ket_r)
    states_r.append(ket_r)
    states_k2.append(ket_k)
    states_k.append(ket_k)

for element in etats3:
    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")

    states_r3.append(ket_r)
    states_r.append(ket_r)
    states_k3.append(ket_k)
    states_k.append(ket_k)

for element in etats4:
    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")

    states_r4.append(ket_r)
    states_r.append(ket_r)
    states_k4.append(ket_k)
    states_k.append(ket_k)

for element in etats:
    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")

    states.append(ket_r)
    states.append(ket_k)


def best_k(states_k, t=1, U=2):

    E = []

    for i in range(4, 11):

        E_i = []

        for etats_k in combinations(states_k, i):

            E0 = ground_energy(list(etats_k), 4, t, U)[0]
            
            E_i.append(E0)

        E.append(min(E_i))

    return E



H, S = get_H_S(4)

ensembles = []
tailles = []

#def optimized_fund_4(states, t=1, U=2):
#
#    E0 = -2.8284
#
# A utiliser pour accelerer (mais moins de precision)
#    
#    if len(states) == 72:
#        
#        E, _, _, vec, _ = optimized_ground_energy(states, H, S, 4, t, U)
#
#        n = []
#
#        for i, element in zip(range(72), vec):
#            if np.isclose(element.real, 0) and np.isclose(element.imag, 0):
#
#                n.append(i)
#    
#        for i in n[::-1]:
#            states.pop(i)
#
#        return optimized_fund_4(states, t, U)
#
#    if len(states) > 30:        
#        for i in range(len(states)):
#
#            new_states = states.copy()
#
#            index = np.random.randint(len(states))
#            new_states.pop(index)
#    
#            E = optimized_ground_energy(new_states, H, S, 4, t, U)[0]
#
#            if E[0] <= E0:
#                return optimized_fund_4(new_states, t, U)
#
#    elif len(states) <= 30:
#        for i in range(len(states)):
#
#            new_states = states.copy()
#
#            new_states.pop(i)
#
#            E = small_optimized_ground_energy(new_states, H, S, 4, t, U)[0]
#
#            if E[0] <= E0:                         
#                return optimized_fund_4(new_states, t, U)
#    
#    if len(states) <= 10 and states not in ensembles:
#
#        ensembles.append(states)
#        tailles.append(len(states))
#
#    return states
#

def optimized_fund_4(states, t=1, U=2):
    
    E0 = -2.8284

    while True:

        if len(states) > 30:
            solver = optimized_ground_energy
            use_random_index = True

        else:
            solver = small_optimized_ground_energy
            use_random_index = False

        removed = False

        for i in range(len(states)):

            new_states = states.copy()
            idx = np.random.randint(len(states)) if use_random_index else i
            new_states.pop(idx)
            E = solver(new_states, H, S, 4, t, U)[0]

            if E[0] <= E0:

                states = new_states
                removed = True
                break

        if not removed:
            break

    if len(states) <= 10 and states not in ensembles:
        ensembles.append(states)
        tailles.append(len(states))

    return states


#for i in range(150):
#
#    debut = time.perf_counter()
#    optimized_fund_4(states)
#    fin = time.perf_counter()
#
#    print(f"{fin-debut:.5f} s")
#
#
#
#print("Energie fondamentale\n")
#print(-2.82843)
#for i in range(len(ensembles)):
#    print(f"\n{ensembles[i]} \n {tailles[i]}")



def minor_polynom(E0, eigvals, eigvec):

    val = 0

    for k in range(len(eigvals)):
        
        term = abs(eigvec[k])**2 * prod((E0 - eigvals[j]) for j in range(len(eigvals)) if j != k)
        val += term

    return val


#ensembles = []
#tailles = []
#visited = set()
#
#def plus_optimized_fund_4(states, eps=1e-10, t=1, U=2):
#
#    key = tuple(sorted(s.number() for s in states))
#
#    if key in visited:
#        return
#
#    visited.add(key)
#
#    E0 = -2.828427124746192
#
#    eigvals, _, _, eigvecs, _ = small_optimized_ground_energy(states, H, S, 4, t, U)
#    
#    print(eigvals[0])
#
#    index_to_flush = []
#
#    for i in range(len(eigvals)):
#        left = minor_polynom(E0 - eps, eigvals, eigvecs[i, :])
#        right = minor_polynom(E0 + eps, eigvals, eigvecs[i, :])
#
#        if left * right < 0:
#            index_to_flush.append(i)
#
#    print(len(index_to_flush))
#
#    for index in index_to_flush:
#        new_states = states.copy()
#        new_states.pop(index)
#        plus_optimized_fund_4(new_states)
#
#    if len(states) <= 10 and states not in ensembles:
#        ensembles.append(states.copy())
#        tailles.append(len(states))
#        print("youppi!")
#        print(states)




#ensembles = []
#tailles = []
#
#def plus_optimized_fund_4(states, eps=1e-10, t=1, U=2):
#
#    E0 = -2.828427124746192
#
#    eigvals, _, _, eigvecs, _ = small_optimized_ground_energy(states, H, S, 4, t, U)
#
#    index_to_flush = []
#
#    for i in range(len(eigvals)):
#
#        left = minor_polynom(E0 - eps, eigvals, eigvecs[i,:])
#        right = minor_polynom(E0 + eps, eigvals, eigvecs[i,:])
#    
#        if left*right < 0:
#            index_to_flush.append(i)
#
#    for index in index_to_flush:
#        
#        new_states = states.copy()
#        new_states.pop(index)
#
#        plus_optimized_fund_4(new_states)
#    
#    if len(states) <= 10 and states not in ensembles:
#
#        ensembles.append(states)
#        tailles.append(len(states))
#
#        print("youppi!")
#        print(states)
#
#    return states
#
#
#
#for i in range(10):
#
#    plus_optimized_fund_4(complete_basis(S, states, 4))
#
#
#print("Energie fondamentale\n")
#print(-2.82843)
#for i in range(len(ensembles)):
#    print(f"\n{ensembles[i]} \n {tailles[i]}")
#


def test_4(t=1, U=2):

    result = best_k(states_k1)

    best = result[:-1]
    fund = result[-1]

    H, S = get_H_S(4, t, U)

    for i in range(1, 8):
        for j in range(1, 8):

            if i + j <= 8:
               
                print(f"\033[1;36m{i} etats en r et {j} etats en k\n\033[0m")

                for etats_r in combinations(states_r, i):

                    states = []
    
                    for e in etats_r:
                        states.append(e)
    
                    for etats_k in combinations(states_k1, j):

                        new_states = states.copy()
        
                        for e in etats_k:
                            new_states.append(e)

                        E0, _, _, _, overfilled = optimized_ground_energy(new_states, H, S, 4, t, U)

                        if set(new_states) == set(ensemble_minimal):

                            print("Energie fondamentale\n")
                            print([s.name for s in new_states])
                            print(round(E0, 5))
                            print("\n")

                        elif set(ensemble_minimal).issubset(new_states):
                            continue

                        elif overfilled:

                            print("Base surcomplete\n")
                            print([s.name for s in new_states])
                            print("\n")
                        
                        elif np.isclose(min(energies), fund):
                        
                            print("Energie fondamentale\n")
                            print([s.name for s in new_states])
                            print(round(fund, 5))
                            print("\n")

                        else:
                            
                            for ind, E in zip(range(9, 3, -1), reversed(best)):

                                if E0 < E and i + j < ind:
                
                                    print(f"Mieux que {ind} etats k\n")
                                    print([s.name for s in new_states])
                                    print(round(E0, 5))
                                    print("\n")
                                    
                                    break

#test_4()



## Test 4-sites : energie fondamentale et etat fondamental


r = [90, 165]
k = [90, 153, 165, 204]



def mixte4(r, k, t=1, U=2, mu=0):
    
    debut = time.perf_counter()

    states = []

    for r in r:
    
        ket_r = Ket(list(map(int, format(r, f'0{8}b'))), "r", str(r) + "r")
        states.append(ket_r)

    for k in k:

        ket_k = Ket(list(map(int, format(k, f'0{8}b'))), "k", str(k) + "k")
        states.append(ket_k)

    E, H, S, omega, overfilled = ground_energy(states, 4, t, U, mu)

    fin = time.perf_counter()


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



#mixte4(r, k)




## Test 6-sites : base mixte


H_r = hamiltonian_r(6)

comps_r = all_components(H_r)

fund_r = []

for c in comps_r:

    if len(c) == (math.factorial(6) ** 2) / (math.factorial(3) ** 4):

        #print(c)

        r_set = c.copy()

        S_sorted = sorted(c)
        H_sub = H_r[np.ix_(S_sorted, S_sorted)]

        eigvals = []

        for val in np.linalg.eigvals(H_sub): 
            eigvals.append(float(round(val.real, 5)))

        for el in eigvals:
            fund_r.append(el)

        states_k = list(c)


        #print(sorted(eigvals))

        break


    
#H_k = hamiltonian_k(6)
#
#comps_k = all_components(H_k)
#
#fund_k = []
#
#for c in comps_k:
#
#    if c.issubset(r_set):
#
#        #print(c)
#
#        S_sorted = sorted(c)
#        H_sub = H_k[np.ix_(S_sorted, S_sorted)]
#
#        eigvals = []
#
#        for val in np.linalg.eigvals(H_sub): 
#            eigvals.append(float(round(val.real, 5)))
#
#        for el in eigvals:
#            fund_k.append(el)
#
#        #print(min(eigvals))
#    
#        if min(eigvals) < -5.4:
#            states_k1 = list(c)
#

#print(sorted(fund_k))


#states = []
#states_k = []
#
#for element in [1386, 2709]:
#    ket_r = Ket(list(map(int, format(element, f'0{12}b'))), "r", str(element) + "r")
#    states.append(ket_r)
#
#for element in states_k1:
#
#    if not element in [3185, 3353, 3241, 2275, 3640]:
#
#        ket_k = Ket(list(map(int, format(element, f'0{12}b'))), "r", str(element) + "r")
#        states_k.append(ket_k)
#
#    if element in [3185, 3353, 3241, 2275, 3640]:
#
#        ket_k = Ket(list(map(int, format(element, f'0{12}b'))), "k", str(element) + "k")
#        states.append(ket_k)
#
#H, S = get_H_S(6, t=1, U=2)
#
#for etats_k in combinations(states_k, 5):
#    
#    new_states = states.copy()
#
#    for el in etats_k:
#        new_states.append(el)
#
#    E0 = optimized_ground_energy(new_states, H, S, 6, 1, 2)[0]
#
#    if E0 < -5.12:
#        
#        print("Energie fondamentale\n")
#        print([s.name for s in new_states])
#        print(round(E0, 5))
#        print("\n")



#states = []
#
#for element in states_k:
#    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
#    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")
#
#    states.append(ket_r)
#    states.append(ket_k)
#

H, S = get_H_S(6)

ensembles = []
tailles = []


#def optimized_fund_6(states, t=1, U=2):
#
#    E0 = -5.4094
#
#    if len(states) > 90:
#        
#        for i in range(len(states)):
#
#            new_states = states.copy()
#
#            index = np.random.randint(len(states))
#            new_states.pop(index)
#    
#            E = optimized_ground_energy(new_states, H, S, 6, t, U)[0]
#
#            if E <= E0:
#                return optimized_fund_6(new_states, t, U)
#
#    elif len(states) <= 90:
#
#        for i in range(len(states)):
#
#            new_states = states.copy()
#
#            new_states.pop(i)
#
#            E = optimized_ground_energy(new_states, H, S, 6, t, U)[0]
#
#            if E <= E0:
#                return optimized_fund_6(new_states, t, U)
#
#    if len(states) <= 70 and states not in ensembles:
#
#        ensembles.append(states)
#        tailles.append(len(states))
#
#    return states


def optimized_fund_6(states, t=1, U=2):

    E0 = -5.4094

    while True:

        if len(states) > 90:
            solver = optimized_ground_energy
            use_random_index = True

        else:
            solver = small_optimized_ground_energy
            use_random_index = False

        removed = False

        for i in range(len(states)):

            new_states = states.copy()
            idx = np.random.randint(len(states)) if use_random_index else i
            new_states.pop(idx)
            E = solver(new_states, H, S, 6, t, U)[0]

            if E[0] <= E0:

                states = new_states
                removed = True
                break

        if not removed:
            break

    if len(states) <= 70 and states not in ensembles:
        ensembles.append(states)
        tailles.append(len(states))

    return states



#for i in range(10):
#    
#    debut = time.perf_counter()
#    optimized_fund_6(states)
#    fin = time.perf_counter()
#
#    print(f"{fin-debut:5f} s")
#
#
#print("Energie fondamentale\n")
#print(-5.40943)
#for i in range(len(ensembles)):
#    print(f"\n{ensembles[i]} \n {tailles[i]}")



def plus_optimized_fund_6(states, t=1, U=2):

    E0 = -5.4094

    if len(states) > 90:
        
        for i in range(len(states)):

            new_states = states.copy()

            index = np.random.randint(len(states))
            new_states.pop(index)
    
            E = optimized_ground_energy(new_states, H, S, 6, t, U)[0]

            if E <= E0:
                return optimized_fund_6(new_states, t, U)

    elif len(states) <= 90:

        for i in range(len(states)):

            new_states = states.copy()

            new_states.pop(i)

            E = optimized_ground_energy(new_states, H, S, 6, t, U)[0]

            if E <= E0:
                return optimized_fund_6(new_states, t, U)

    if len(states) <= 70 and states not in ensembles:

        ensembles.append(states)
        tailles.append(len(states))

    return states




## Test 6-sites : energie fondamentale et etat fondamental


r = [1386, 2709]
k = [3185, 3353, 3241, 3640, 2275, 1253, 1652, 2674, 2296, 1619, 3619, 2387, 2420, 3365]


def mixte6(r, k, t=1, U=2):
    
    debut = time.perf_counter()

    states = []

    for r in r:
    
        ket_r = Ket(list(map(int, format(r, f'0{12}b'))), "r", str(r) + "r")
        states.append(ket_r)

    for k in k:

        ket_k = Ket(list(map(int, format(k, f'0{12}b'))), "k", str(k) + "k")
        states.append(ket_k)

    E, H, S, omega, overfilled = ground_energy(states, 6, t, U)

    fin = time.perf_counter()

    print(states)
    if overfilled == True:
        print("\n")
        print("BASE SURCOMPLETE")
    print("\n")
    print("Energie fondamentale : ", round(E[0], 5))
    print("\n")
    print("H = \n", H)
    print("\n")
    print("S = \n", S)
    print("\n")
    print("Etat fondamental : ", omega[0])

    print("\n")
    print(f"Temps d'exécution total : {fin - debut:.6f} s")

    return E[0], H, S, omega[0]

#mixte6(r, k)



## Test etat mixte 2 sites

etats = np.arange(4**(2*2))

states = []

compteur = 0

for element in etats:
    
    lst = list(map(int, format(element, f'0{8}b')))

    if sum(lst[:2*2]) != 1 or sum(lst[2*2:]) != 1:
        continue
    
    ket = Ket(lst, None, str(element), compteur=compteur)
    states.append(ket)

    compteur += 1


def m_test_2(t=1, U=2):

    for etats in combinations(states, 2):

        H, S = m_get_H_S(2, t, U)
        
        E0 = m_optimized_ground_energy(list(etats), H, S, 2, t, U)[0][0]
        
        if E0 < -1.23:

            print([s.name for s in etats])
            print(round(E0, 5))


#m_test_2()



## Test etat mixte 4 sites

etats = np.arange(4**(2*4))

states = []

for element in etats:
    
    lst = list(map(int, format(element, f'0{16}b')))

    if sum(lst[:2*4]) != 2 or sum(lst[2*4:]) != 2:
        continue
    
    ket = Ket(lst, None, str(element))
    states.append(ket)


def m_test_4(t=1, U=2):

    debut = time.perf_counter()

    H, S = m_get_H_S(4, t, U)

    fin = time.perf_counter()

    print(f"H/S computed in {fin-debut:.5f} s")

    for etats in combinations(states, 3):

        debut = time.perf_counter()
        
        E0 = m_optimized_ground_energy(list(etats), H, S, 4, t, U)[0]
        
        if E0 < -2.82:

            print([s.name for s in etats])
            print(round(E0, 5))

        fin = time.perf_counter()

        print(f"{fin-debut:.5f} s")

#m_test_4()



## Fonction de Green non orthonormee (construction de N sous-espaces excites mu)



# Etats de la base de depart

#r = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]
#k = []

#r = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]
#k = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]

#r = [90, 165]
#k = [90, 165, 153, 204]

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

spin = 0        # Spin up --> 0  /  Spin down --> 1
basis = "k"



def etats_decomposes(N):
    
    ensemble = []

    for positions in combinations(range(N), N//2):

        liste = [0] * N
        
        for i in positions:
            liste[i] = 1

        ensemble.append(liste)

    resultat = []

    for a in ensemble:
        for b in ensemble:
            resultat.append(a + b)

    return resultat



def Ht_mu(state1, state2, N, mu, t, excitation):

    diff = [b - a for a, b in zip(state1, state2)]

    if diff.count(1) != 1 or diff.count(-1) != 1:
        return 0

    i = diff.index(1)
    j = diff.index(-1)
    
    if excitation == "e":
        if j != mu:
            return 0
    
    elif excitation == "h":
        if i != mu:
            return 0

    if abs(i - j) == 1:
        exp = 0

    elif abs(i - j) == N - 1:
        exp = N // 2 - 1

    else:
        return 0

    return t * (-1) ** exp



def S_plus_r(states, N, mu, spin):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    S_plus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                S_plus[i, j] = int(i == j) * int(states[i].vector[mu + N*spin] == 0)

            elif states[i].type == "r" and states[j].type == "k":

                S_plus[i, j] = int(states[i].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "r":

                S_plus[i, j] = int(states[j].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "k":

                S_plus[i, j] = 0

                etats = etats_decomposes(N)

                for l in range(len(etats)):

                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2)]
                    s2 = S[int(''.join(map(str, etats[l])), 2), states[j].number()]

                    S_plus[i, j] += int(etats[l][mu + N*spin] == 0) * s1 * s2

    return S_plus




def H_plus_r(states, N, mu, spin, t, U, mu1):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    H_plus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                delta = int(i == j)

                term1 = Ht_mu(states[i].vector, states[j].vector, N, mu + N*spin, t, "e")
                term2 = U * int(states[j].vector[mu + N*int(spin == 0)] == 1) * delta
                term3 = -mu1 * delta
                term4 = ham_r(states[i].vector, states[j].vector, N, t, U, mu1)

                H_plus[i, j] = term1 + int(states[j].vector[mu + N*spin] == 0) * (term2 + term3 + term4)
                     

            elif states[i].type == "r" and states[j].type == "k":

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

            elif states[i].type == "k" and states[j].type == "r":

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

            elif states[i].type == "k" and states[j].type == "k":

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

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    S_moins = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                S_moins[i, j] = int(i == j) * int(states[i].vector[mu + N*spin] == 1)

            elif states[i].type == "r" and states[j].type == "k":

                S_moins[i, j] = int(states[i].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "r":

                S_moins[i, j] = int(states[j].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "k":

                S_moins[i, j] = 0

                etats = etats_decomposes(N)

                for l in range(len(etats)):

                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2)]
                    s2 = S[int(''.join(map(str, etats[l])), 2), states[j].number()]
                    
                    S_moins[i, j] += int(etats[l][mu + N*spin] == 1) * s1 * s2

    return S_moins




def H_moins_r(states, N, mu, spin, t, U, mu1):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    H_moins = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                delta = int(i == j)

                term1 = Ht_mu(states[i].vector, states[j].vector, N, mu + N*spin, t, "h")
                term2 = -U * int(states[j].vector[mu + N*int(spin == 0)] == 1) * delta
                term3 = mu1 * delta
                term4 = ham_r(states[i].vector, states[j].vector, N, t, U, mu1)

                H_moins[i, j] = term1 + int(states[j].vector[mu + N*spin] == 1) * (term2 + term3 + term4)
                     

            elif states[i].type == "r" and states[j].type == "k":

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

            elif states[i].type == "k" and states[j].type == "r":

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

            elif states[i].type == "k" and states[j].type == "k":

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
    
    diff = [b - a for a, b in zip(state1, state2)]

    if diff[:N].count(1) != 1 or diff[:N].count(-1) != 1 or diff[N:].count(1) != 1 or diff[N:].count(-1) != 1:
        return 0

    i = diff[:N].index(1)
    j = diff[:N].index(-1)
    k = diff[N:].index(1) + N
    l = diff[N:].index(-1) + N

    state_up = state2[:N]
    state_down = state2[N:]


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


    if (k-l) % N != (j-i) % N:
        return 0
    
    exp = 0

    if i < j:
        exp += sum(state_up[:j]) + sum(state_up[:i]) - 1

    elif i > j:
        exp += sum(state_up[:j]) + sum(state_up[:i])

    if k < l:
        exp += sum(state_down[:l-N]) + sum(state_down[:k-N]) - 1

    elif k > l:
        exp += sum(state_down[:l-N]) + sum(state_down[:k-N])

    return -U/N * (-1) ** exp




def S_plus_k(states, N, mu, spin):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    S_plus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                S_plus[i, j] = 0

                etats = etats_decomposes(N)

                for l in range(len(etats)):

                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2) + 4**N]
                    s2 = S[int(''.join(map(str, etats[l])), 2) + 4**N, states[j].number()]

                    S_plus[i, j] += int(etats[l][mu + N*spin] == 0) * s1 * s2

            elif states[i].type == "r" and states[j].type == "k":

                S_plus[i, j] = int(states[j].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "r":

                S_plus[i, j] = int(states[i].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "k":

                S_plus[i, j] = int(states[i].vector[mu + N*spin] == 0) * int(i == j)

    return S_plus




def H_plus_k(states, N, mu, spin, t, U, mu1):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    H_plus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

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

            elif states[i].type == "r" and states[j].type == "k":

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

            elif states[i].type == "k" and states[j].type == "r":

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

            elif states[i].type == "k" and states[j].type == "k":

                delta = int(i == j)

                term1 = U/N * sum(states[j].vector[N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                term2 = HU_mu(states[i].vector, states[j].vector, N, mu, spin, U, "e")
                term3 = -2*t * np.cos((2*np.pi*mu)/N) * delta
                term4 = -mu1 * delta
                term5 = ham_k(states[i].vector, states[j].vector, N, t, U, mu1)

                H_plus[i, j] = term2 + int(states[j].vector[mu + N*spin] == 0) * (term1 + term3 + term4 + term5)

    return H_plus




def S_moins_k(states, N, mu, spin):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    S_moins = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                S_moins[i, j] = 0

                etats = etats_decomposes(N)

                for l in range(len(etats)):

                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2) + 4**N]
                    s2 = S[int(''.join(map(str, etats[l])), 2) + 4**N, states[j].number()]

                    S_moins[i, j] += int(etats[l][mu + N*spin] == 1) * s1 * s2

            elif states[i].type == "r" and states[j].type == "k":

                S_moins[i, j] = int(states[j].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "r":

                S_moins[i, j] = int(states[i].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "k":

                S_moins[i, j] = int(states[i].vector[mu + N*spin] == 1) * int(i == j)

    return S_moins




def H_moins_k(states, N, mu, spin, t, U, mu1):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    H_moins = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

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
                     

            elif states[i].type == "r" and states[j].type == "k":

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


            elif states[i].type == "k" and states[j].type == "r":

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

            elif states[i].type == "k" and states[j].type == "k":

                delta = int(i == j)

                term1 = -U/N * sum(states[j].vector[N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                term2 = HU_mu(states[i].vector, states[j].vector, N, mu, spin, U, "h")
                term3 = 2*t * np.cos((2*np.pi*mu)/N) * delta
                term4 = mu1 * delta
                term5 = ham_k(states[i].vector, states[j].vector, N, t, U, mu1)

                H_moins[i, j] = term2 + int(states[j].vector[mu + N*spin] == 1) * (term1 + term3 + term4 + term5)


    return H_moins




def mix_green(N, spin, t, U, mu, basis):

    states = []

    for etat_r in r:

        ket_r = Ket(list(map(int, format(etat_r, f'0{2*N}b'))), "r", str(etat_r) + "r")
        states.append(ket_r)

    for etat_k in k:

        ket_k = Ket(list(map(int, format(etat_k, f'0{2*N}b'))), "k", str(etat_k) + "k")
        states.append(ket_k)

    E, H, S, omega, overfilled = ground_energy(states, N, t, U, mu)

    E0 = E[0]
    Omega = omega[0]

    w = np.linspace(-10, 10, 2000)
    dos = np.zeros(len(w))

    for i in range(N):

        debut = time.perf_counter()
        
        if basis == "r":
            S_plus = S_plus_r(states, N, i, spin)
            H_plus = H_plus_r(states, N, i, spin, t, U, mu)

        elif basis == "k":
            S_plus = S_plus_k(states, N, i, spin)
            H_plus = H_plus_k(states, N, i, spin, t, U, mu)

        fin = time.perf_counter()

        print(f"Matrix creation : {fin-debut:.6f} s")

        debut = time.perf_counter()

        eig_S_plus = np.linalg.eigh(S_plus)[0]

        if np.any(np.isclose(eig_S_plus, 0, atol=1e-6)):
        
            e_eigvals, e_eigenvectors = gen_diagonalization(H_plus, S_plus)       

        else:

            e_eigvals, e_eigenvectors = sp.linalg.eigh(H_plus, S_plus)

       
        if basis == "r":
            S_moins = S_moins_r(states, N, i, spin)
            H_moins = H_moins_r(states, N, i, spin, t, U, mu)

        if basis == "k":
            S_moins = S_moins_k(states, N, i, spin)
            H_moins = H_moins_k(states, N, i, spin, t, U, mu)

        eig_S_moins = np.linalg.eigh(S_moins)[0]

        if np.any(np.isclose(eig_S_moins, 0, atol=1e-6)):

            h_eigvals, h_eigenvectors = gen_diagonalization(H_moins, S_moins)       

        else:

            h_eigvals, h_eigenvectors = sp.linalg.eigh(H_moins, S_moins)

        fin = time.perf_counter()

        print(f"Diagonalization : {fin-debut:.6f} s")

        debut = time.perf_counter()

        Q_e = Omega.conj().T @ S_plus @ e_eigenvectors
        Q_h = Omega.conj().T @ S_moins @ h_eigenvectors

        g = green(w, Q_e, Q_h, e_eigvals, h_eigvals, E0, N)  

        fin = time.perf_counter()

        print(f"Green : {fin-debut:.6f} s")      

        dos += g

    #plt.plot(w, g + i, "r")
    #plt.show()

    return w, dos



def green(x, Q_e, Q_h, e_eigs, h_eigs, E0, N, eta=0.05j):
    
    green_function = np.zeros(len(x))

    for i in range(len(x)):
        
        value = 0
        
        for j in range(len(Q_e)):
            
            value += np.abs(Q_e[j])**2/(x[i] + eta + E0 - e_eigs[j]) 
            
        for j in range(len(Q_h)):

            value += np.abs(Q_h[j])**2/(x[i] + eta - E0 + h_eigs[j])
        
        green_function[i] = value.imag/(-N*np.pi)
    
    return green_function


w, dos = mix_green(N, spin, t, U, mu, basis)

plt.plot(w, dos, "r")

plt.xlabel(r"$\omega$")
plt.ylabel(r"$n(\omega)$")
plt.title(rf"#sites = {N}    $U = {U}$    $\mu = {mu}$    $N = {N}$    $S_z = 0$")
plt.grid()
plt.show()


# Graphique des DOS (r, k et exact) (ne fonctionne que pour N = 4, t = 1 et U = 2, 4, 8, ou 12)

#w_k, dos_k = mix_green(N, spin, t, U, mu, "k")
#w_r, dos_r = mix_green(N, spin, t, U, mu, "r")
#
#
#data = np.loadtxt(f"dos_u{U}", skiprows=2)
#
#x = data[:, 0]
#y1 = data[:, 1]
#y2 = data[:, 2]
#
#plt.plot(x, y1 + y2, "black", label="exact")
#plt.plot(w_r, dos_r, "r", label="r")
#plt.plot(w_k, dos_k, "b", label="k")
#
#plt.xlabel(r'$\omega$')
#plt.ylabel(r'$n(\omega)$')
#
#plt.title(rf"#sites = {N}    $U = {U}$    $\mu = {mu}$    $N = {N}$    $S_z = 0$")
#
#plt.legend()
#plt.grid()
#plt.show()


# Graphique de 4 DOS (r, k et exact) (U = 2, 4, 8, 12) en une figure

#U_values = [2, 4, 8, 12]
#
#fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
#
#for ax, U in zip(axes, U_values):
#
#    w_k, dos_k = mix_green(N, spin, t, U, U/2, "k")
#    w_r, dos_r = mix_green(N, spin, t, U, U/2, "r")
#
#    data = np.loadtxt(f"dos_u{U}", skiprows=2)
#
#    x = data[:, 0]
#    y1 = data[:, 1]
#    y2 = data[:, 2]
#
#    ax.plot(x, y1 + y2, "black", label="exact")
#    ax.plot(w_r, dos_r, "r", label="r")
#    ax.plot(w_k, dos_k, "b", label="k")
#
#    ax.set_ylabel(r"$n(\omega)$")
#    ax.set_title(rf"$U = {U}$", loc="left")
#
#    ax.legend()
#    ax.grid()
#
#
#axes[-1].set_xlabel(r"$\omega$")
#
#fig.suptitle(rf"#sites = {N}    $t = {t}$    $\mu = U / 2$    $S_z = 0$", fontsize=14)
#
#plt.tight_layout()
#plt.show()




## Fonction de Green non orthonormee (construction d'un seul sous-espace excite)



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

spin = 0        # Spin up --> 0  /  Spin down --> 1
basis = "k"



def etats_decomposes(N):
    
    ensemble = []

    for positions in combinations(range(N), N//2):

        liste = [0] * N
        
        for i in positions:
            liste[i] = 1

        ensemble.append(liste)

    resultat = []

    for a in ensemble:
        for b in ensemble:
            resultat.append(a + b)

    return resultat



def Ht_mu(state1, state2, N, mu, t, excitation):

    diff = [b - a for a, b in zip(state1, state2)]

    if diff.count(1) != 1 or diff.count(-1) != 1:
        return 0

    i = diff.index(1)
    j = diff.index(-1)
    
    if excitation == "e":
        if j != mu:
            return 0
    
    elif excitation == "h":
        if i != mu:
            return 0

    if abs(i - j) == 1:
        exp = 0

    elif abs(i - j) == N - 1:
        exp = N // 2 - 1

    else:
        return 0

    return t * (-1) ** exp



def S_plus_r(states, N, mu, spin):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    S_plus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                S_plus[i, j] = int(i == j) * int(states[i].vector[mu + N*spin] == 0)

            elif states[i].type == "r" and states[j].type == "k":

                S_plus[i, j] = int(states[i].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "r":

                S_plus[i, j] = int(states[j].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "k":

                S_plus[i, j] = 0

                etats = etats_decomposes(N)

                for l in range(len(etats)):

                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2)]
                    s2 = S[int(''.join(map(str, etats[l])), 2), states[j].number()]

                    S_plus[i, j] += int(etats[l][mu + N*spin] == 0) * s1 * s2

    return S_plus




def H_plus_r(states, N, mu, spin, t, U, mu1):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    H_plus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                delta = int(i == j)

                term1 = Ht_mu(states[i].vector, states[j].vector, N, mu + N*spin, t, "e")
                term2 = U * int(states[j].vector[mu + N*int(spin == 0)] == 1) * delta
                term3 = -mu1 * delta
                term4 = ham_r(states[i].vector, states[j].vector, N, t, U, mu1)

                H_plus[i, j] = term1 + int(states[j].vector[mu + N*spin] == 0) * (term2 + term3 + term4)
                     

            elif states[i].type == "r" and states[j].type == "k":

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

            elif states[i].type == "k" and states[j].type == "r":

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

            elif states[i].type == "k" and states[j].type == "k":

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

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    S_moins = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                S_moins[i, j] = int(i == j) * int(states[i].vector[mu + N*spin] == 1)

            elif states[i].type == "r" and states[j].type == "k":

                S_moins[i, j] = int(states[i].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "r":

                S_moins[i, j] = int(states[j].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "k":

                S_moins[i, j] = 0

                etats = etats_decomposes(N)

                for l in range(len(etats)):

                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2)]
                    s2 = S[int(''.join(map(str, etats[l])), 2), states[j].number()]
                    
                    S_moins[i, j] += int(etats[l][mu + N*spin] == 1) * s1 * s2

    return S_moins




def H_moins_r(states, N, mu, spin, t, U, mu1):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    H_moins = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                delta = int(i == j)

                term1 = Ht_mu(states[i].vector, states[j].vector, N, mu + N*spin, t, "h")
                term2 = -U * int(states[j].vector[mu + N*int(spin == 0)] == 1) * delta
                term3 = mu1 * delta
                term4 = ham_r(states[i].vector, states[j].vector, N, t, U, mu1)

                H_moins[i, j] = term1 + int(states[j].vector[mu + N*spin] == 1) * (term2 + term3 + term4)
                     

            elif states[i].type == "r" and states[j].type == "k":

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

            elif states[i].type == "k" and states[j].type == "r":

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

            elif states[i].type == "k" and states[j].type == "k":

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
    
    diff = [b - a for a, b in zip(state1, state2)]

    if diff[:N].count(1) != 1 or diff[:N].count(-1) != 1 or diff[N:].count(1) != 1 or diff[N:].count(-1) != 1:
        return 0

    i = diff[:N].index(1)
    j = diff[:N].index(-1)
    k = diff[N:].index(1) + N
    l = diff[N:].index(-1) + N

    state_up = state2[:N]
    state_down = state2[N:]


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


    if (k-l) % N != (j-i) % N:
        return 0
    
    exp = 0

    if i < j:
        exp += sum(state_up[:j]) + sum(state_up[:i]) - 1

    elif i > j:
        exp += sum(state_up[:j]) + sum(state_up[:i])

    if k < l:
        exp += sum(state_down[:l-N]) + sum(state_down[:k-N]) - 1

    elif k > l:
        exp += sum(state_down[:l-N]) + sum(state_down[:k-N])

    return -U/N * (-1) ** exp




def S_plus_k(states, N, mu, spin):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    S_plus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                S_plus[i, j] = 0

                etats = etats_decomposes(N)

                for l in range(len(etats)):

                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2) + 4**N]
                    s2 = S[int(''.join(map(str, etats[l])), 2) + 4**N, states[j].number()]

                    S_plus[i, j] += int(etats[l][mu + N*spin] == 0) * s1 * s2

            elif states[i].type == "r" and states[j].type == "k":

                S_plus[i, j] = int(states[j].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "r":

                S_plus[i, j] = int(states[i].vector[mu + N*spin] == 0) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "k":

                S_plus[i, j] = int(states[i].vector[mu + N*spin] == 0) * int(i == j)

    return S_plus




def H_plus_k(states, N, mu, spin, t, U, mu1):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    H_plus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

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

            elif states[i].type == "r" and states[j].type == "k":

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

            elif states[i].type == "k" and states[j].type == "r":

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

            elif states[i].type == "k" and states[j].type == "k":

                delta = int(i == j)

                term1 = U/N * sum(states[j].vector[N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                term2 = HU_mu(states[i].vector, states[j].vector, N, mu, spin, U, "e")
                term3 = -2*t * np.cos((2*np.pi*mu)/N) * delta
                term4 = -mu1 * delta
                term5 = ham_k(states[i].vector, states[j].vector, N, t, U, mu1)

                H_plus[i, j] = term2 + int(states[j].vector[mu + N*spin] == 0) * (term1 + term3 + term4 + term5)

    return H_plus




def S_moins_k(states, N, mu, spin):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    S_moins = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

                S_moins[i, j] = 0

                etats = etats_decomposes(N)

                for l in range(len(etats)):

                    s1 = S[states[i].number(), int(''.join(map(str, etats[l])), 2) + 4**N]
                    s2 = S[int(''.join(map(str, etats[l])), 2) + 4**N, states[j].number()]

                    S_moins[i, j] += int(etats[l][mu + N*spin] == 1) * s1 * s2

            elif states[i].type == "r" and states[j].type == "k":

                S_moins[i, j] = int(states[j].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "r":

                S_moins[i, j] = int(states[i].vector[mu + N*spin] == 1) * S[states[i].number(), states[j].number()]

            elif states[i].type == "k" and states[j].type == "k":

                S_moins[i, j] = int(states[i].vector[mu + N*spin] == 1) * int(i == j)

    return S_moins




def H_moins_k(states, N, mu, spin, t, U, mu1):

    dim = len(states)

    S = np.load(f"S_{N}.npy")

    H_moins = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):

            if states[i].type == "r" and states[j].type == "r":

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
                     

            elif states[i].type == "r" and states[j].type == "k":

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


            elif states[i].type == "k" and states[j].type == "r":

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

            elif states[i].type == "k" and states[j].type == "k":

                delta = int(i == j)

                term1 = -U/N * sum(states[j].vector[N*int(spin == 0): N*(2-int(spin == 1))]) * delta
                term2 = HU_mu(states[i].vector, states[j].vector, N, mu, spin, U, "h")
                term3 = 2*t * np.cos((2*np.pi*mu)/N) * delta
                term4 = mu1 * delta
                term5 = ham_k(states[i].vector, states[j].vector, N, t, U, mu1)

                H_moins[i, j] = term2 + int(states[j].vector[mu + N*spin] == 1) * (term1 + term3 + term4 + term5)


    return H_moins




def mix_green(N, spin, t, U, mu, basis):

    states = []

    for etat_r in r:

        ket_r = Ket(list(map(int, format(etat_r, f'0{2*N}b'))), "r", str(etat_r) + "r")
        states.append(ket_r)

    for etat_k in k:

        ket_k = Ket(list(map(int, format(etat_k, f'0{2*N}b'))), "k", str(etat_k) + "k")
        states.append(ket_k)

    E, H, S, omega, overfilled = ground_energy(states, N, t, U, mu)

    E0 = E[0]
    Omega = omega[0]

    w = np.linspace(-10, 10, 2000)
    dos = np.zeros(len(w))

    for i in range(N):

        debut = time.perf_counter()
        
        if basis == "r":
            S_plus = S_plus_r(states, N, i, spin)
            H_plus = H_plus_r(states, N, i, spin, t, U, mu)

        elif basis == "k":
            S_plus = S_plus_k(states, N, i, spin)
            H_plus = H_plus_k(states, N, i, spin, t, U, mu)

        fin = time.perf_counter()

        print(f"Matrix creation : {fin-debut:.6f} s")

        debut = time.perf_counter()

        eig_S_plus = np.linalg.eigh(S_plus)[0]

        if np.any(np.isclose(eig_S_plus, 0, atol=1e-6)):
        
            e_eigvals, e_eigenvectors = gen_diagonalization(H_plus, S_plus)       

        else:

            e_eigvals, e_eigenvectors = sp.linalg.eigh(H_plus, S_plus)

       
        if basis == "r":
            S_moins = S_moins_r(states, N, i, spin)
            H_moins = H_moins_r(states, N, i, spin, t, U, mu)

        if basis == "k":
            S_moins = S_moins_k(states, N, i, spin)
            H_moins = H_moins_k(states, N, i, spin, t, U, mu)

        eig_S_moins = np.linalg.eigh(S_moins)[0]

        if np.any(np.isclose(eig_S_moins, 0, atol=1e-6)):

            h_eigvals, h_eigenvectors = gen_diagonalization(H_moins, S_moins)       

        else:

            h_eigvals, h_eigenvectors = sp.linalg.eigh(H_moins, S_moins)

        fin = time.perf_counter()

        print(f"Diagonalization : {fin-debut:.6f} s")

        debut = time.perf_counter()

        Q_e = Omega.conj().T @ S_plus @ e_eigenvectors
        Q_h = Omega.conj().T @ S_moins @ h_eigenvectors

        g = green(w, Q_e, Q_h, e_eigvals, h_eigvals, E0, N)  

        fin = time.perf_counter()

        print(f"Green : {fin-debut:.6f} s")      

        dos += g

    #plt.plot(w, g + i, "r")
    #plt.show()

    return w, dos



def green(x, Q_e, Q_h, e_eigs, h_eigs, E0, N, eta=0.05j):
    
    green_function = np.zeros(len(x))

    for i in range(len(x)):
        
        value = 0
        
        for j in range(len(Q_e)):
            
            value += np.abs(Q_e[j])**2/(x[i] + eta + E0 - e_eigs[j]) 
            
        for j in range(len(Q_h)):

            value += np.abs(Q_h[j])**2/(x[i] + eta - E0 + h_eigs[j])
        
        green_function[i] = value.imag/(-N*np.pi)
    
    return green_function


w, dos = mix_green(N, spin, t, U, mu, basis)

plt.plot(w, dos, "r")

plt.xlabel(r"$\omega$")
plt.ylabel(r"$n(\omega)$")
plt.title(rf"#sites = {N}    $U = {U}$    $\mu = {mu}$    $N = {N}$    $S_z = 0$")
plt.grid()
plt.show()


# Graphique des DOS (r, k et exact) (ne fonctionne que pour N = 4, t = 1 et U = 2, 4, 8, ou 12)

#w_k, dos_k = mix_green(N, spin, t, U, mu, "k")
#w_r, dos_r = mix_green(N, spin, t, U, mu, "r")
#
#
#data = np.loadtxt(f"dos_u{U}", skiprows=2)
#
#x = data[:, 0]
#y1 = data[:, 1]
#y2 = data[:, 2]
#
#plt.plot(x, y1 + y2, "black", label="exact")
#plt.plot(w_r, dos_r, "r", label="r")
#plt.plot(w_k, dos_k, "b", label="k")
#
#plt.xlabel(r'$\omega$')
#plt.ylabel(r'$n(\omega)$')
#
#plt.title(rf"#sites = {N}    $U = {U}$    $\mu = {mu}$    $N = {N}$    $S_z = 0$")
#
#plt.legend()
#plt.grid()
#plt.show()


# Graphique de 4 DOS (r, k et exact) (U = 2, 4, 8, 12) en une figure

#U_values = [2, 4, 8, 12]
#
#fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
#
#for ax, U in zip(axes, U_values):
#
#    w_k, dos_k = mix_green(N, spin, t, U, U/2, "k")
#    w_r, dos_r = mix_green(N, spin, t, U, U/2, "r")
#
#    data = np.loadtxt(f"dos_u{U}", skiprows=2)
#
#    x = data[:, 0]
#    y1 = data[:, 1]
#    y2 = data[:, 2]
#
#    ax.plot(x, y1 + y2, "black", label="exact")
#    ax.plot(w_r, dos_r, "r", label="r")
#    ax.plot(w_k, dos_k, "b", label="k")
#
#    ax.set_ylabel(r"$n(\omega)$")
#    ax.set_title(rf"$U = {U}$", loc="left")
#
#    ax.legend()
#    ax.grid()
#
#
#axes[-1].set_xlabel(r"$\omega$")
#
#fig.suptitle(rf"#sites = {N}    $t = {t}$    $\mu = U / 2$    $S_z = 0$", fontsize=14)
#
#plt.tight_layout()
#plt.show()














