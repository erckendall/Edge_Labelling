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

# Initialise starting configuration for face/edge matrix
mat_in_face = np.zeros((n_face, n_edge))
mat_in_face[0, 0] = mat_in_face[0, 4] = mat_in_face[0, 7] = mat_in_face[0, 11] = 1
mat_in_face[1, 1] = mat_in_face[1, 9] = mat_in_face[1, 10] = mat_in_face[1, 11] = 1
mat_in_face[2, 2] = mat_in_face[2, 6] = mat_in_face[2, 8] = mat_in_face[2, 9] = 1
mat_in_face[3, 3] = mat_in_face[3, 4] = mat_in_face[3, 5] = mat_in_face[3, 6] = 1
mat_in_face[4, 5] = mat_in_face[4, 7] = mat_in_face[4, 8] = mat_in_face[4, 10] = 1

# Define initial set of cycles using edge labels
cycles_in = [[5,4,3,6], [1,10,9,11]]

# Set the number of edge addition operations to perform
iterations = 10

# Function to determine which edges are not in cycles
def func_empties(mat, cycles):
    ons = []
    empty = []
    for i in cycles:
        for j in i:
            ons.append(j)
    for k in range(mat.shape[1]):
        if k not in ons:
            empty.append(k)
    if len(empty) != n_edge/3:
        print 'Empties incorrect!!!', '(length: ', len(empty), ', should be:', n_edge/3, ')'
        exit()
    return empty

# Function to check that resulting vertex/edge matrix has the correct structure
def func_check_validity(mat):
    valid = True
    rows = np.shape(mat)[0]
    cols = np.shape(mat)[1]
    if 1.5 * rows != cols:
        valid = False
    for i in range(rows):
        sum = np.sum(mat[i])
        if sum != 3.:
            valid = False
    for i in range(cols):
        sum = np.sum(mat[:,i])
        if sum != 2.:
            valid = False
    return valid

# Function to arrange cycle edges in order
def func_ordering(mat, cycles):
    cycs = cycles
    for i in range(len(cycs)):
        if len(cycs[i]) % 2 != 0:
            print 'uneven cycle: ', cycs[i]
            exit()
    ordered_cycles = []
    for i in range(len(cycs)):
        ln = len(cycs[i])
        cyc = cycs[i]
        ordered_cyc = []
        ordered_cyc.append(cyc[0])
        occ_verts = []
        for j in range(mat.shape[0]):
            if mat[j, cyc[0]] == 1:
                occ_verts.append(j)
        targ = occ_verts[0]
        cyc.remove(cyc[0])
        complete = False
        while complete == False:
            for q in cyc:
                if mat[targ, q] == 1:
                    ordered_cyc.append(q)
                    cyc.remove(q)
                    for x in range(mat.shape[0]):
                        if mat[x, q] == 1 and x != targ:
                            new_target = x
                            if new_target == occ_verts[1]:
                                complete = True
            if len(ordered_cyc) == ln:
                complete = True
            targ = new_target
        ordered_cycles.append(ordered_cyc)
    for x in range(len(mat[0, :])):
        occ = 0
        for i in ordered_cycles:
            if x in i:
                occ += 1
            if occ > 1:
                ordered_cycles.remove(i)
    return ordered_cycles

# Function to create all on/off combinations. Input to this function is the output of the ordering function
def func_combos(ordering):
    sets = []
    for set in ordering:
        first = []
        second = []
        for i in range(len(set)):
            if i % 2 == 0:
                first.append(set[i])
            else:
                second.append(set[i])
        sets.append([first, second])
    combs = []
    for subset in its.product(*sets):
        combs.append(subset)
    return combs

# Function to create 3 edge groups for proper edge labelling
def func_labels(combs, empties, n_edge):
    labellings = []
    for i in combs:
        groups = [empties,[],[]]
        for j in i:
            for k in j:
                groups[1].append(k)
        for j in range(n_edge):
            if j not in groups[0] and j not in groups[1]:
                groups[2].append(j)
        labellings.append(groups)
    return labellings

# Function to create new on/off groups
def func_new_ons(mat, combos, empties):
    for i in combos:
        len_tot = 0
        for j in i:
            len_tot += len(j)
        if (len_tot + len(empties)) % 2 != 0:
            print 'uneven total number of edges on cycles'
            exit()
    ons = []
    for i in range(len(combos)):
        on = []
        for j in range(np.shape(mat)[1]):
            for x in range(len(combos[i])):
                if j in combos[i][x]:
                   on.append(j)
        for k in empties:
            on.append(k)
        ons.append(on)
    return ons

# Function to separate on/off groups into cycles
def func_more_cycles(new_ons, mat):
    all_cycs = []
    for q in range(len(new_ons)):
        verts_tot = []
        for i in new_ons[q]:
            verts = []
            for j in range(mat.shape[0]):
                if mat[j][i] == 1:
                    verts.append(j)
            verts_tot.append(verts)
        set_lst = [set(i) for i in verts_tot]
        full_cover = False
        full_lst = []
        while full_cover == False:
            lst = set_lst[0]
            complete = False
            while complete == False:
                chk = 0
                to_rem = 'nil'
                for i in range(len(set_lst)):
                    if list(set_lst[i] & lst) != []:
                        comp1 = list(set_lst[i])
                        comp2 = list(lst)
                        lst = set(comp1 + comp2)
                        to_rem = set_lst[i]
                        chk += 1
                if to_rem in set_lst:
                    set_lst.remove(to_rem)
                if chk == 0:
                    complete = True
            full_lst.append(list(lst))
            ln = 0
            for i in full_lst:
                ln += len(i)
            if ln == len(new_ons[q]):
                full_cover = True
        all_cycs.append(full_lst)
    all_edge_sets = []
    for x in range(len(new_ons)):
        edge_sets = []
        for i in all_cycs[x]:
            edge_set = []
            for j in i:
                for k in new_ons[x]:
                    if mat[j][k] == 1:
                        edge_set.append(k)
            edge_set = list(set(edge_set))
            edge_sets.append(edge_set)
        all_edge_sets.append(edge_sets)
    return all_edge_sets

# Main loop to determine all cycle coverings of the map from the first cycle input
def func_main_algo(mat, cycles):
    validity = func_check_validity(mat)
    if validity == False:
        print "Invalid matrix representation!"
        exit()
    ordered_cycle_list = []
    complete = False
    to_do = [cycles]
    labels = []
    while complete == False:
        for cycs in to_do:
            empties = func_empties(mat, cycs)
            ord_cyc = func_ordering(mat, cycs)
            if ord_cyc not in ordered_cycle_list:
                ordered_cycle_list.append(ord_cyc)
                combos = func_combos(ord_cyc)
                labs = func_labels(combos, empties, n_edge)
                for t in labs:
                    labels.append(t)
                on_combos = func_new_ons(mat, combos, empties)
                new_cycles = func_more_cycles(on_combos, mat)
                for i in new_cycles:
                    sm = 0
                    for j in i:
                        if len(j) % 2 != 0:
                            print 'New cycles uneven'
                            exit()
                        for k in j:
                            sm += 1
                    if sm != 2 * n_edge/3:
                        print 'incomplete cycle generation'
                        exit()
                    to_do.append(i)
            to_do.remove(cycs)
            if to_do == []:
                complete = True
    return [ordered_cycle_list, labels]

# Randomly choose a face to segment with new edge, and existing edges along which new edge terminates
def func_choose_edges(mat_face):
    rge = range(mat_face.shape[0])
    face = random.choice(rge)
    edges = []
    for i in range(mat_face.shape[1]):
        if mat_face[face, i] == 1:
            edges.append(i)
    edge_0 = random.choice(edges)
    edge_1 = random.choice(edges)
    return [face, edges, edge_0, edge_1]

# Compute the order of edges around the relevant face
def func_order_face(mat, edges):
    ordered_edges = [edges[0]]
    to_order = edges
    to_order.remove(edges[0])
    complete = False
    occ_verts = []
    for i in range(np.shape(mat)[0]):
        if mat[i, edges[0]] == 1:
            occ_verts.append(i)
    target = occ_verts[0]
    while complete == False:
        for j in to_order:
            if mat[target, j] == 1:
                ordered_edges.append(j)
                to_order.remove(j)
                for p in range(mat.shape[0]):
                    if mat[p, j] == 1 and p != target:
                        target_new = p
                        if target_new == occ_verts[1]:
                            complete = True
        target = target_new
    return ordered_edges

# Redefine vertex/edge matrix - two cases depending on whether edge_0 and edge_1 are the same
def func_new_v_e_mat(mat, edge_0, edge_1, n_vert, n_edge):
    mat_in_new = np.zeros((n_vert + 2, n_edge + 3))
    mat_in_new[n_vert][n_edge + 2] = 1
    mat_in_new[n_vert + 1][n_edge + 2] = 1
    verts_0 = []
    verts_1 = []
    if edge_0 != edge_1:
        for i in range(mat.shape[1]):
            if i != edge_0 and i != edge_1:
                for j in range(mat.shape[0]):
                    mat_in_new[j][i] = mat[j][i]
            elif i == edge_0:
                for j in range(mat.shape[0]):
                    if mat[j, i] == 1:
                        verts_0.append(j)
                mat_in_new[verts_0[0]][i] = 1
                mat_in_new[n_vert][i] = 1
            elif i == edge_1:
                for j in range(mat.shape[0]):
                    if mat[j, i] == 1:
                        verts_1.append(j)
                mat_in_new[verts_1[1]][i] = 1
                mat_in_new[n_vert+1][i] = 1
        mat_in_new[verts_0[1]][n_edge] = 1
        mat_in_new[n_vert][n_edge] = 1
        mat_in_new[verts_1[0]][n_edge + 1] = 1
        mat_in_new[n_vert + 1][n_edge + 1] = 1
    else:
        for i in range(mat.shape[1]):
            if i != edge_0:
                for j in range(mat.shape[0]):
                    mat_in_new[j][i] = mat[j][i]
            elif i == edge_0:
                for j in range(mat.shape[0]):
                    if mat[j, i] == 1:
                        verts_0.append(j)
                mat_in_new[n_vert][i] = 1
                mat_in_new[n_vert + 1][i] = 1
        mat_in_new[verts_0[0]][n_edge] = 1
        mat_in_new[n_vert][n_edge] = 1
        mat_in_new[verts_0[1]][n_edge + 1] = 1
        mat_in_new[n_vert + 1][n_edge + 1] = 1
    return [mat_in_new, verts_0, verts_1]

# Choose a covering of cycles from which to build new cycles after edge addition - Hamiltons preferred
def func_choose_cycle(prev_cycles, prev_hamiltons, edge_0, edge_1):
    chosen = False
    while chosen == False:
        for i in prev_hamiltons:
            for j in i:
                if edge_0 in j and edge_1 in j:
                    to_use = i
                    chosen = True
        if chosen == False:
            print 'None of the Hamiltonian cycles suitable for edge addition'
            for i in prev_cycles:
                for j in i:
                    if edge_0 in j and edge_1 in j:
                        to_use = i
                        chosen = True
            if chosen == False:
                print 'No suitable map covering found!'
                exit()
    return to_use

# Redifine the hamilton (or other) cycle(s)
def func_new_cycles(edge_0, to_use, new_mat):
    new_ham = to_use
    for i in to_use:
        if edge_0 in i:
            i.append(n_edge)
            i.append(n_edge + 1)
    sm = 0
    for i in new_ham:
        for j in i:
            sm += 1
    if sm != 2 * (n_edge + 3) / 3:
        print 'Cycle after edge addition incorrect:'
        exit()
    cycs = new_ham
    ordered_ham = []
    for i in range(len(cycs)):
        ln = len(cycs[i])
        cyc = cycs[i]
        ordered_cyc = []
        ordered_cyc.append(cyc[0])
        occ_verts = []
        for j in range(new_mat.shape[0]):
            if new_mat[j, cyc[0]] == 1:
                occ_verts.append(j)
        targ = occ_verts[0]
        cyc.remove(cyc[0])
        complete = False
        while complete == False:
            for q in cyc:
                if new_mat[targ, q] == 1:
                    ordered_cyc.append(q)
                    cyc.remove(q)
                    for x in range(new_mat.shape[0]):
                        if new_mat[x, q] == 1 and x != targ:
                            new_target = x
                            if new_target == occ_verts[1]:
                                complete = True
            if len(ordered_cyc) == ln:
                complete = True
            targ = new_target
        ordered_ham.append(ordered_cyc)
    for x in range(len(new_mat[0, :])):
        occ = 0
        for i in ordered_ham:
            if x in i:
                occ += 1
            if occ > 1:
                ordered_ham.remove(i)
    return ordered_ham

# Function to redefine edge/face matrix
def func_new_e_f_mat(n_edge, n_face, n_vert, edges, mat_in_new, face, edge_0, edge_1, mat_face):
    new_faces = [n_edge + 2, n_edge + 1, n_edge] + edges
    face_0 = [n_edge]
    for i in range(mat_in_new.shape[0]):
        if mat_in_new[i][n_edge] == 1 and i != n_vert:
            vertex = i
    complete = False
    new_faces.remove(n_edge)
    while complete == False:
        for j in new_faces:
            if mat_in_new[vertex][j] == 1:
                face_0.append(j)
                new_faces.remove(j)
                if mat_in_new[n_vert + 1][j] == 1:
                    complete = True
                else:
                    for k in range(mat_in_new.shape[0]):
                        if mat_in_new[k][j] == 1 and k != vertex:
                            to_vert = k
                    vertex = to_vert
    face_0.append(n_edge + 2)
    face_1 = new_faces
    mat_in_face_new = np.zeros((n_face + 1, n_edge + 3))
    for i in range(n_edge):
        for j in range(n_face):
            mat_in_face_new[j,i] = mat_face[j,i]
        if i in face_1:
            mat_in_face_new[face, i] = 0
            mat_in_face_new[n_face, i] = 1
    for i in range(n_edge, n_edge + 3):
        if i in face_0:
            mat_in_face_new[face, i] = 1
        if i in face_1:
            mat_in_face_new[n_face, i] = 1
        for j in range(n_face):
            if j != face and mat_face[j, edge_0] == 1 and i == n_edge:
                mat_in_face_new[j, i] = mat_face[j, edge_0]
            if j != face and mat_face[j, edge_1] == 1 and i == n_edge + 1:
                mat_in_face_new[j, i] = mat_face[j, edge_1]
    return mat_in_face_new

# Function to remove cyclically permuted duplicates from list of cycles
def func_remove_duplicates(ordered_cycle_list, mat):
    ordered_cycle_list2 = ordered_cycle_list
    for x in range(len(ordered_cycle_list)):
        for q in range(len(ordered_cycle_list[x])):
            ordered_cycle_list2[x][q] = sorted(ordered_cycle_list[x][q])
    ordered_cycle_list3 = []
    for i in range(len(ordered_cycle_list2)):
        ordered_cycle_list3.append(np.array(sorted(ordered_cycle_list2[i], key=lambda x: x[0])).tolist())
    ordered_cycle_list3 = [tuple(tuple(i) for i in t) for t in ordered_cycle_list3]
    ordered_cycle_list3 = list(set(ordered_cycle_list3))
    ordered_cycle_list3 = [list(list(cyc) for cyc in elem) for elem in ordered_cycle_list3]
    final_ordered_cycle_list = []
    for i in ordered_cycle_list3:
        final_ordered_cycle_list.append(func_ordering(mat, i))
    return final_ordered_cycle_list

# Function to remove duplicate labellings under a/b/c exchange
def func_duplicate_labels(labels):
    labels_2 = []
    for i in labels:
        groups = []
        for j in i:
            groups.append(sorted(j))
        labels_2.append(groups)
    labels_3 = []
    for i in labels_2:
        lst = np.sort(i, 0).tolist()
        labels_3.append(lst)
    labels_3.sort()
    final = list(labels_3 for labels_3, _ in its.groupby(labels_3))
    return final

# Main loop: Computes cycle coverings and adds an additional edge
mat = mat_in
mat_face = mat_in_face
cycles = cycles_in
n = 0
while n < iterations:
    print '\n Iteration', n, ':'
    # print 'Current vertex/edge matrix: \n', mat
    # print 'Current face/edge matrix: \n', mat_face
    out = func_main_algo(mat, cycles)
    ordered_cycle_list = out[0]
    labels = out[1]
    labels_final = func_duplicate_labels(labels)
    final_ordered_cycle_list = func_remove_duplicates(ordered_cycle_list, mat)
    hamiltons = []
    for i in final_ordered_cycle_list:
        if len(i) == 1:
            hamiltons.append(i)

    # Print final results (note than one can here add lines to print the full cycles, labellings, etc.)
    if len(hamiltons) == 0:
        print 'No Hamilton cycles found!'
        exit()
    print 'Total number of CCGs: ', len(final_ordered_cycle_list)
    print 'Number of Hamiltonian cycles: ', len(hamiltons)
    print 'Number of distinct proper labellings: ', len(labels_final)

    # Add edge and generate new map
    prev_hamiltons = hamiltons
    prev_cycles = []
    for i in final_ordered_cycle_list:
        if i not in hamiltons:
            prev_cycles.append(i)
    new_line = func_choose_edges(mat_face)
    face = new_line[0]
    edges = new_line[1]
    edges_2 = np.copy(edges).tolist()
    edge_0 = new_line[2]
    edge_1 = new_line[3]
    print 'EDGE ADDITION STEP: \n', 'Face chosen:', face, ', Edge(s) chosen:', edge_0, edge_1
    ordered_edges = func_order_face(mat, edges)
    redef = func_new_v_e_mat(mat, edge_0, edge_1, n_vert, n_edge)
    mat_in_new = redef[0]
    mat_in_face_new = func_new_e_f_mat(n_edge, n_face, n_vert, edges_2, mat_in_new, face, edge_0, edge_1, mat_face)
    verts_0 = redef[1]
    verts_1 = redef[2]
    to_use = func_choose_cycle(prev_cycles, prev_hamiltons, edge_0, edge_1)
    new_ham = func_new_cycles(edge_0, to_use, mat_in_new)
    print 'New CCG:', new_ham

    if n == iterations - 1:
        print '\n Final vertex/edge matrix: \n', mat_in_new
        print 'Final face/edge matrix: \n', mat_in_face_new

    # Set new inputs for iteration
    mat = mat_in_new
    mat_face = mat_in_face_new
    cycles = new_ham
    n_vert += 2
    n_edge += 3
    n_face += 1
    n += 1










