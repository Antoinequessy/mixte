import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from itertools import combinations

np.set_printoptions(precision=5, linewidth=150, suppress=True)


class Ket:
    def __init__(self, vector, typ, name=None):
        self.vector = np.array(vector)
        self.name = name
        self.type = typ

    def __repr__(self):
        return self.name if self.name is not None else "Ket"


def determinant_1kr(N):
    """
    Calcule le déterminant de la matrice de
    transformation de base k vers r pour un
    système à N sites comportant 1 électron.
    Retourne la suite {1, -1, -i, -i, -1, 1, i, i}
    *** Pour les N pairs : {-1, -i, 1, i}
    *** Pour les N impairs : {1, -i, -1, i}
    """
    
    m = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for k in range(N):

            m[i, k] = np.exp((2j*i*k*np.pi)/N)

    m = (1/np.sqrt(N)) * m

    det = np.linalg.det(m)

    return det


def determinant_1rk(N):
    """
    Calcule le déterminant de la matrice de
    transformation de base r vers k pour un
    système à N sites comportant 1 électron.
    Retourne la suite {1, -1, i, i, -1, 1, -i, -i}
    *** Pour les N pairs : {-1, i, 1, -i}
    *** Pour les N impairs : {1, i, -1, -i}
    """
    
    m = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for k in range(N):

            m[i, k] = np.exp((-2j*i*k*np.pi)/N)

    m = (1/np.sqrt(N)) * m

    det = np.linalg.det(m)

    return det


def metrique_rk(N):

    dim = 4**N
    matrice = np.zeros((dim, dim), dtype=complex)

    occ = ((np.arange(dim)[:, None] >> np.arange(2*N)) & 1)[:, ::-1]

    indices = np.arange(2*N)
    F = np.exp(-2j * np.pi * np.outer(indices, indices) / N)

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


def metrique_m(N):

    dim = N
    matrice = np.zeros((dim, dim), dtype=complex)
    
    return matrice


def hamiltonian_r(N, t=1, U=2):

    dim = 4 ** N

    H = np.zeros((dim, dim), dtype=float)

    etats = []

    for i in range(dim):

        etat = []

        for j in range(2*N):
            etat.append(0)

        compteur = 0

        for j in range(2*N):
            
            if i - compteur >= 2**(2*N - (j+1)):

                etat[j] = 1

                compteur += 2**(2*N - (j+1))

        etats.append(etat)

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

    etats = []

    for i in range(dim):

        etat = []

        for j in range(2*N):
            etat.append(0)

        compteur = 0

        for j in range(2*N):
            
            if i - compteur >= 2**(2*N - (j+1)):

                etat[j] = 1

                compteur += 2**(2*N - (j+1))

        etats.append(etat)

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



def ham_k(state1, state2, N, t=1, U=2):

    if np.array_equal(state1, state2):

        state_up = state1[:N]
        state_down = state1[N:]
    
        n_up = sum(state_up)
        n_down = sum(state_down)

        somme = 0
        
        for i in range(N):

            somme += np.cos((2*np.pi*i) / N) * (state_up[i] + state_down[i])
        
        H = -2*t * somme + U/N * n_up * n_down

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



def ham_r(state1, state2, N, t=1, U=2):

    if np.array_equal(state1, state2):

        state_up = state1[:N]
        state_down = state1[N:]

        n_U = 0

        for i in range(N):
                       
            if state_up[i] == 1 and state_down[i] == 1:

                n_U += 1

        H = U * n_U

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



def ham_m(order, state1, state2, N, t=1, U=2):

    """
    Ordre rk ou kr des états state1 et state2
    """

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
            

    H = met_m(order, state1, state2, N) * (H_t + H_U)
    
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


def met_m(order, state1, state2, N):

    state1_up = state1[:N]
    state1_down = state1[N:]
    
    n_up1 = sum(state1_up)
    n_down1 = sum(state1_down)

    state2_up = state2[:N]
    state2_down = state2[N:]
    
    n_up2 = sum(state2_up)
    n_down2 = sum(state2_down)

    if n_up1 != n_up2 or n_down1 != n_down2:

        S = 0

        return S


    matrice_up = np.zeros((n_up1, n_up1), dtype=complex)
    matrice_down = np.zeros((n_down1, n_down1), dtype=complex)

    r_up = []
    k_up = []
    r_down = []
    k_down = []

    for i in range(N):

        if order == "rk":

            if state1_up[i] == 1:

                r_up.append(i)

            if state2_up[i] == 1:

                k_up.append((2*np.pi*i)/N)

            if state1_down[i] == 1:

                r_down.append(i)

            if state2_down[i] == 1:

                k_down.append((2*np.pi*i)/N)

        elif order == "kr":

            if state1_up[i] == 1:

                k_up.append((2*np.pi*i)/N)

            if state2_up[i] == 1:

                r_up.append(i)

            if state1_down[i] == 1:

                k_down.append((2*np.pi*i)/N)

            if state2_down[i] == 1:

                r_down.append(i)
        

    for i in range(n_up1):
        for j in range(n_up1):

            if order == "rk":

                matrice_up[i, j] = np.exp(1j*k_up[i]*r_up[j])

            elif order == "kr":

                matrice_up[i, j] = np.exp(-1j*k_up[i]*r_up[j])

    for i in range(n_down1):
        for j in range(n_down1):

            if order == "rk":

                matrice_down[i, j] = np.exp(1j*k_down[i]*r_down[j])

            elif order == "kr":

                matrice_down[i, j] = np.exp(-1j*k_down[i]*r_down[j])

    S = N ** (-1/2*(n_up1+n_down1)) * np.linalg.det(matrice_up) * np.linalg.det(matrice_down)

    return S


def ground_energy(states, N, t=1, U=2):

    dim = len(states)

    H = np.zeros((dim, dim), dtype=complex)
    S = np.zeros((dim, dim), dtype=complex)

    for i in range(len(states)):
        for j in range(len(states)):

            if states[i].type == "r" and states[j].type == "r":
                
                H[i, j] = ham_r(states[i].vector, states[j].vector, N, t, U)
                S[i, j] = met_r(states[i].vector, states[j].vector, N)

            elif states[i].type == "r" and states[j].type == "k":
                
                H[i, j] = ham_m("rk", states[i].vector, states[j].vector, N, t, U)
                S[i, j] = met_m("rk", states[i].vector, states[j].vector, N)

            elif states[i].type == "k" and states[j].type == "r":
                
                H[i, j] = ham_m("kr", states[i].vector, states[j].vector, N, t, U)
                S[i, j] = met_m("kr", states[i].vector, states[j].vector, N)

            elif states[i].type == "k" and states[j].type == "k":
                
                H[i, j] = ham_k(states[i].vector, states[j].vector, N, t, U)
                S[i, j] = met_k(states[i].vector, states[j].vector, N)

    eigenvalues, eigenvectors = sp.linalg.eigh(H, S)

    return eigenvalues, H, S



def fundamental_energy(states, t=0.5, U=2):

    key = lambda ket: tuple(ket.vector.tolist())

    hamiltonian = {
        key(ket_5r): Ket(U * ket_5r.vector - t * (ket_6r.vector + ket_9r.vector), "Hket_5r"),
        key(ket_6r): Ket(-t * (ket_5r.vector + ket_10r.vector), "Hket_6r"),
        key(ket_9r): Ket(-t * (ket_5r.vector + ket_10r.vector), "Hket_9r"),
        key(ket_10r): Ket(U * ket_10r.vector - t * (ket_6r.vector + ket_9r.vector), "Hket_10r"),

        key(ket_5k): Ket((2*t + U/2) * ket_5k.vector + U/2 * ket_10k.vector, "Hket_5k"),
        key(ket_6k): Ket(U/2 * (ket_6k.vector + ket_9k.vector), "Hket_6k"),
        key(ket_9k): Ket(U/2 * (ket_6k.vector + ket_9k.vector), "Hket_9k"),
        key(ket_10k): Ket((-2*t + U/2) * ket_10k.vector + U/2 * ket_5k.vector, "Hket_10k"),
    }

    dim = len(states)
    H = np.zeros((dim, dim), dtype=float)
    S = np.zeros((dim, dim), dtype=float)

    for i in range(dim):
        for j in range(dim):
            H[i, j] = np.dot(states[i].vector,
                             hamiltonian[key(states[j])].vector)

    for i in range(dim):
        for j in range(dim):
            S[i, j] = np.dot(states[i].vector, states[j].vector)

    eigenvalues, eigenvectors = sp.linalg.eigh(H, S)

    return eigenvalues


ket_5r = Ket([1, 0, 0, 0], "ket_5r")
ket_6r = Ket([0, 1, 0, 0], "ket_6r")
ket_9r = Ket([0, 0, 1, 0], "ket_9r")
ket_10r = Ket([0, 0, 0, 1], "ket_10r")

ket_5k = Ket(1/2 * (ket_5r.vector - ket_6r.vector - ket_9r.vector + ket_10r.vector), "ket_5k")
ket_6k = Ket(1/2 * (ket_5r.vector + ket_6r.vector - ket_9r.vector - ket_10r.vector), "ket_6k")
ket_9k = Ket(1/2 * (ket_5r.vector - ket_6r.vector + ket_9r.vector - ket_10r.vector), "ket_9k")
ket_10k = Ket(1/2 * (ket_5r.vector + ket_6r.vector + ket_9r.vector + ket_10r.vector), "ket_10k")

##states_1 = [ket_5r, ket_6r, ket_9r, ket_10r]
##
##E_1 = fundamental_energy(states_1)
##
##E_1 = [float(round(e, 5)) for e in E_1]
##
##print(E_1)
##
##states_2 = [ket_5r, ket_10k, ket_10r]
##
##E_2 = fundamental_energy(states_2)
##
##E_2 = [float(round(e, 5)) for e in E_2]
##
##print(E_2)

##states = [ket_5r, ket_6r, ket_9r, ket_10r, ket_5k, ket_6k, ket_9k, ket_10k]
##
##for a, b, c in combinations(states, 3):
##    etats = [a, b, c]
##
##    E = fundamental_energy(etats)
##    E = [float(round(e, 5)) for e in E_2]
##
##    print([s.name for s in etats])
##    print(E)

##M = hamiltonian_r(4)
##
##for i in range(len(M)):
##
##    row = M[i]
##
##    for j in range(len(row)):
##
##        if row[j] != 0:
##
##            print("ket", i,". ket",  j, " = ", row[j])


from collections import deque

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


##H_r = hamiltonian_r(4)
##
##comps_r = all_components(H_r)
##
##fund_r = []
##
##for c in comps_r:
##
##    print(c)
##
##    S_sorted = sorted(c)
##    H_sub = H_r[np.ix_(S_sorted, S_sorted)]
##
##    eigvals = []
##
##    for val in np.linalg.eigvals(H_sub): 
##        eigvals.append(float(round(val.real, 5)))
##
##    for el in eigvals:
##        fund_r.append(el)
##
##    print(eigvals)
##
##print(sorted(fund_r))
##
##    
##H_k = hamiltonian_k(4)
##
##comps_k = all_components(H_k)
##
##fund_k = []
##
##for c in comps_k:
##
##    print(c)
##
##    S_sorted = sorted(c)
##    H_sub = H_k[np.ix_(S_sorted, S_sorted)]
##
##    eigvals = []
##
##    for val in np.linalg.eigvals(H_sub): 
##        eigvals.append(float(round(val.real, 5)))
##
##    for el in eigvals:
##        fund_k.append(el)
##
##    print(eigvals)
##
##print(sorted(fund_k))

etats = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]

states_r = []
states_k = []

states = []

for element in etats:
    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")

    states.append(ket_r)
    states_r.append(ket_r)
    states.append(ket_k)
    states_k.append(ket_k)


def test(n, t=1, U=2):

    ensemble_test = states_r

    ensemble_test[2] = states_k[2]

    E0 = min(ground_energy(ensemble_test, 4, t, U)[0])

##    for etats in combinations(states, n):
##        E = ground_energy(list(etats), 4, t, U)
##        E = [float(round(e, 5)) for e in E]
##
##        if min(E) == E0:

##            print([s.name for s in etats])

    print(ground_energy(ensemble_test, 4, t, U)[2])

print(test(1))




