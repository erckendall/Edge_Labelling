#This code constitutes a much faster way to construct the full set of CCGs than the previous technique

import numpy as np
import itertools as its
import random

# Setting appropriate matrix dimension for planar cubic map
n_vert = 8
n_edge = int(1.5 * n_vert)
n_face = int(n_vert/2. + 1)

# Initialise starting configuration for vertex/edge matrix
mat_in = np.zeros((n_vert, n_edge))
mat_in[0, 0] = mat_in[1, 0] = 1
mat_in[1, 1] = mat_in[2, 1] = 1
mat_in[2, 2] = mat_in[3, 2] = 1
mat_in[0, 3] = mat_in[3, 3] = 1
mat_in[0, 4] = mat_in[7, 4] = 1
mat_in[6, 5] = mat_in[7, 5] = 1
mat_in[3, 6] = mat_in[6, 6] = 1
mat_in[5, 7] = mat_in[7, 7] = 1
mat_in[4, 8] = mat_in[6, 8] = 1
mat_in[2, 9] = mat_in[4, 9] = 1
mat_in[4, 10] = mat_in[5, 10] = 1
mat_in[1, 11] = mat_in[5, 11] = 1


# First generate all possible combinations of 2/3 of the total number of edges
lst = []
for i in range(n_edge):
    lst.append(i)
combos = []
for subset in its.combinations(lst,  n_vert):
    combos.append(subset)


# Now filter these to check that each edge in the set shares a vertex with exactly two others in the set
filtered = []
for comb in combos:
    bad = False
    for j in comb:
        tot_share = 0
        for k in range(n_vert):
            shared = 0
            for l in comb:
                if l != j and mat_in[k, l] == 1 and mat_in[k, j] == 1:
                    shared += 1
                    tot_share += 1
            if shared == 2:
                bad = True
        if tot_share != 2:
            bad = True
    if bad == False:
        filtered.append(comb)

# Now separate the sets into mutually exclusive CCGs
CCGs_unchecked = []
for filt in filtered:
    comb = []
    for i in filt:
        comb.append(i)
    CCGs = []
    while len(comb) > 0:
        for i in comb:
            grp = [i]
            comb.remove(i)
            incomplete = True
            while incomplete == True:
                cnt = 0
                for l in comb:
                    connected = False
                    for k in range(n_vert):
                        for test in grp:
                            if mat_in[k, l] == 1 and mat_in[k, test] == 1:
                                connected = True
                                cnt += 1
                    if connected == True:
                        grp.append(l)
                        comb.remove(l)
                if cnt == 0 or len(comb) == 0:
                    incomplete = False
            CCGs.append(grp)
    CCGs_unchecked.append(CCGs)


# Finally, disregard any CCGs which possess uneven cycles
CCGs_checked = []
for CCGs in CCGs_unchecked:
    bad = False
    for CCG in CCGs:
        if len(CCG) % 2 != 0:
            bad = True
    if bad == False:
        CCGs_checked.append(CCGs)




