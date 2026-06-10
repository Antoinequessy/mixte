import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from itertools import combinations
from collections import deque

np.set_printoptions(precision=5, linewidth=150, suppress=True)


class Ket:
    def __init__(self, vector, typ, name=None):
        self.vector = np.array(vector)
        self.name = name
        self.type = typ

    def __repr__(self):
        return self.name if self.name is not None else "Ket"



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

                k_up.append(i)

            if state2_up[i] == 1:

                r_up.append((2*np.pi*i)/N)

            if state1_down[i] == 1:

                k_down.append(i)

            if state2_down[i] == 1:

                r_down.append((2*np.pi*i)/N)

        elif order == "kr":

            if state1_up[i] == 1:

                r_up.append((2*np.pi*i)/N)

            if state2_up[i] == 1:

                k_up.append(i)

            if state1_down[i] == 1:

                r_down.append((2*np.pi*i)/N)

            if state2_down[i] == 1:

                k_down.append(i)
        

    for i in range(n_up1):
        for j in range(n_up1):

            if order == "rk":

                matrice_up[i, j] = np.exp(1j*r_up[i]*k_up[j])

            elif order == "kr":

                matrice_up[i, j] = np.exp(-1j*k_up[i]*r_up[j])

    for i in range(n_down1):
        for j in range(n_down1):

            if order == "rk":

                matrice_down[i, j] = np.exp(1j*r_down[i]*k_down[j])

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




## Test 4-sites : sous-espaces r/k et energies fondamentales associees


#H_r = hamiltonian_r(4)
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
#    print(eigvals)
#
#print(sorted(fund_r))
#
#    
#H_k = hamiltonian_k(4)
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
#    print(eigvals)
#
#print(sorted(fund_k))



## Test 4-sites : etat mixte de 6 etats (2r, 4k)


#etats = [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204]
#
#etats1 = [51, 60, 90, 102, 105, 150, 153, 165, 195, 204]
#etats2 = [53, 83, 92, 106, 154, 166, 169, 197]
#etats3 = [54, 57, 85, 99, 108, 147, 156,  170, 198, 201]
#etats4 = [58, 86, 89, 101, 149, 163, 172, 202]
#
#states_r1 = []
#states_r2 = []
#states_r3 = []
#states_r4 = []
#states_k1 = []
#states_k2 = []
#states_k3 = []
#states_k4 = []
#
#states = []
#
#for element in etats1:
#    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
#    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")
#
#    states_r1.append(ket_r)
#    states_k1.append(ket_k)
#
#for element in etats2:
#    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
#    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")
#
#    states_r2.append(ket_r)
#    states_k2.append(ket_k)
#
#for element in etats3:
#    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
#    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")
#
#    states_r3.append(ket_r)
#    states_k3.append(ket_k)
#
#for element in etats4:
#    ket_r = Ket(list(map(int, format(element, f'0{8}b'))), "r", str(element) + "r")
#    ket_k = Ket(list(map(int, format(element, f'0{8}b'))), "k", str(element) + "k")
#
#    states_r4.append(ket_r)
#    states_k4.append(ket_k)
#
#
#
#def test_4(t=1, U=2):
#
#    for etats_r in combinations(states_r1, 2):
#
#        states = []
#    
#        for e in etats_r:
#            states.append(e)
#    
#        for etats_k in combinations(states_k1, 4):
#
#            new_states = states.copy()
#        
#            for e in etats_k:
#                new_states.append(e)
#
#            E0 = min(ground_energy(new_states, 4, 1, 2)[0])
#
#            if E0 < -2.8:
#
#                print([s.name for s in new_states])
#                print(round(E0, 5))
#
#
#test_4()



## Test 2-sites : 3 etats


#etats = [5, 6, 9, 10]
#
#states_r = []
#states_k = []
#states = []
#
#for element in etats:
#    ket_r = Ket(list(map(int, format(element, f'0{4}b'))), "r", str(element) + "r")
#    ket_k = Ket(list(map(int, format(element, f'0{4}b'))), "k", str(element) + "k")
#
#    states.append(ket_r)
#    states_r.append(ket_r)
#    states.append(ket_k)
#    states_k.append(ket_k)
#
#
#def test_2(t=1, U=2):
#
#    for etats in combinations(states, 3):
#        
#        E0 = min(ground_energy(list(etats), 2, t, U)[0])
#
#        if E0 < -3.12:
#
#            print([s.name for s in etats])
#            print(round(E0, 5))
#
#
#test_2() 

