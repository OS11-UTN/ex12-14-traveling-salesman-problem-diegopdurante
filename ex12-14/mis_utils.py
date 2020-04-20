import numpy as np



def get_col_names(nn, node_names = None, as_numpy = False, sep = ""):
    str_col_names = ""
    idxs = np.argwhere(nn != 0)

    str_col_names = ""
    np_col_names = np.array(())
    
    for i,j in idxs:
        if node_names is not None:
            str_node_name = node_names[i] + sep + node_names[j]
            str_col_names = str_col_names + str_node_name + str('-')
            np_col_names = np.append(np_col_names, np.array((str_node_name)))
        else:
            str_node_name = str(i+1) + sep + str(j+1)
            str_col_names = str_col_names + str_node_name + "-"
            
    str_col_names = str_col_names[:-1]
    
    if as_numpy == False:
        retval = str_col_names
    else:
        retval = np_col_names

    return retval



def nn2na(nn, node_names = None, show_results = False):

    idxs = np.argwhere(nn > 0) # Brinda una matriz de las combinaciones entre i y j posibles
    if show_results == True:
        print(idxs)

    na = np.zeros((nn.shape[0], idxs.shape[0]), dtype=int)

    for i,j in enumerate(idxs):
        na[j[0],i]=1
        na[j[1], i]=-1
        
    if show_results == True:
        str_col_names = get_col_names(nn, node_names)

        print("Input: \n", nn)
        print()
        
        print('Column names: ' + str_col_names)
        print()
        
        print("Output: \n", na)
        print()
    
    arc_idxs = [(arc[0], arc[1]) for arc in idxs]

    return na, arc_idxs



def get_selected_arcs(res, nan_names = None):
    res_int = res.round().astype(int)
    
    arcs = []
    
    for idx, i in enumerate(np.argwhere(res_int == 1)):
#         print(idx, nan_names[i])
        arcs.append(nan_names[i][0])
        
    return arcs


def get_prec(nn, idx):
    row_search = nn[idx,:]
    return np.argwhere(row_search==1)[:,0]


def dijkstra(NN, C, node_names, initial_node = 0):
    len_nodes = NN.shape[0]

    nan_names = get_col_names(NN, node_names, as_numpy=True)

    weights = np.full((len_nodes), np.inf)
    prec = np.zeros(len_nodes, dtype='l')
    non_explored = np.ones(len_nodes)

    weights[initial_node] = 0

    unexplored = []
    for i in range(len_nodes):
        unexplored.append(i)


    actual_node = initial_node
    it = 0

    final_table_weights = [weights]
    final_table_prec = [prec]

    while(np.sum(non_explored)>0):
        print()

        unexplored_idx = []
        unexplored_weights = []

        # Hago una lista de los nodos no explorados
        for i in np.argwhere(non_explored==1):
            unexplored_idx.append((i[0]))
            unexplored_weights.append((weights[i][0]))
            
        print("Sin explorar: ", unexplored_idx)
        print("Pesos de los nodos sin explorar: ", unexplored_weights)
        
        if len(unexplored_idx) == 0:
            break
            
        # Tomo el que tiene menor peso
        head_node = unexplored_idx[np.argmin(unexplored_weights)]

        non_explored[head_node]=0
            
        # Exploro cada uno de los sucesores del nodo elegido
        print("Nodo elegido ", node_names[head_node])
        for sucesor in get_prec(NN, head_node):
            
            idx_c = np.argwhere(nan_names == str(node_names[head_node]) + str(node_names[sucesor]))[0][0]
            
            print(node_names[head_node], " => ", node_names[sucesor], " con un costo de ", C[idx_c])
            
            potential_weight = weights[head_node] + C[idx_c]
            
            if potential_weight<weights[sucesor]:
                weights[sucesor] = potential_weight
                prec[sucesor] = head_node
                
        print(weights)
        print(prec)
        
        final_table_weights = np.append([final_table_weights][0], [weights], axis = 0)
        final_table_prec = np.append([final_table_prec][0], [prec], axis = 0)
        
        print("Tabla de pesos:\n", final_table_weights)
        print("Tabla de precedentes:\n", final_table_prec)
        print()

    return weights, prec



def dfs_from_NN(NN, _stack = None, _actual_path = None, start = -1, end = -1):
    
    debug_param = False

    stack = []
    actual_path = []

    if _stack != None:
        stack = _stack.copy()

    if _actual_path != None:
        actual_path = _actual_path.copy()

    def get_connections(NN, visited, frm, end = -1):
#         visited[end]=0
        return np.argwhere((NN[frm] * (visited == 0))>0)[:,0]
    
#     actual_path = []
#     stack = [] # Lista, acepta append y pop
    costs = []
    
    visited = np.zeros((NN.shape[0]))
    
    if start == -1:
        start = 0
    if end == -1:
        end = NN.shape[0] - 1

    if debug_param == True:
        print(NN[end])        
        print(get_connections(NN, visited, 0))
    
    actual = start
    
    actual_path.append(start)
        
    while visited.sum()!= NN.shape[0]:
        
        if actual == end:
            if debug_param == True:
                print("Se encontró una ruta: ", actual_path)
            return actual_path, visited, stack #, costs
        
        next_possibes = get_connections(NN, visited, actual)
        if debug_param == True:
            print("Desde: ", actual, " - Posibles siguientes: ", next_possibes.transpose(), " con los siguientes recorridos ", actual_path)
        
        # Si no hay más caminos desde el actual, unstackear
        if(next_possibes.shape[0] == 0):
            
            if actual == start:
                if debug_param == True:
                    print("No hay más caminos por explorar")
                return None, None, None
#             if actual != end:
            visited[actual] = 1
            actual = stack.pop()
            actual_path.pop()

        else:
            stack.append(actual)
#             costs.append(NN[actual, next_possibes[0]])
            actual = next_possibes[0]
            actual_path.append(actual)

def _get_all_paths_DFS(NN,  # Son los enlaces NN
                    target_node,   # Objetivo (número)
                    current_stack,  # Stack? lista de python
                    visited_nodes,    # Nodos visitados en el camino actual lista de python
                    found_paths, # Lista de rutas hasta acá lista de python
                    ):
    # if type(current_stack) == 'int':
    #     current_stack = [current_stack]
    #     print(current_stack)
    # print(type(current_stack))
    last_node = current_stack[-1]

    if last_node == target_node:
        found_paths.append(current_stack.copy())

    else:
        for neighbor in np.argwhere(NN[last_node,:]>0):
            if neighbor[0] not in visited_nodes:
                current_stack.append(neighbor[0])
                visited_nodes.append(neighbor[0])
                _get_all_paths_DFS(NN, target_node, current_stack, visited_nodes, found_paths)
                visited_nodes.remove(neighbor[0])
                current_stack.pop()
    
    return found_paths

def get_all_paths_DFS(NN,  # Son los enlaces NN
                    target_node,   # Objetivo (número)
                    current_stack):  # Stack? lista de python
    return _get_all_paths_DFS(NN, target_node, [current_stack], [], list())

# No anda, no se usa más
def dfs_all_paths_NN(_NN, _stack = None, _actual_path = None, start = -1, end = -1, debug_info = False):

    NN = _NN.copy()    
    stack = []
    actual_path = []

    if _stack != None:
        stack = _stack.copy()

    if _actual_path != None:
        actual_path = _actual_path.copy()

    def get_connections(NN, visited, frm, end = -1):
        # Parche para que no marque coo recorrido, quiero todos los caminos que llegan desde s hasta t,
        # por ende a t lo podría recorrer varias veces
        visited[end]=0 
        
        return np.argwhere((NN[frm] * (visited == 0))>0)[:,0]

    visited = np.zeros((NN.shape[0]))
    
    if start == -1:
        start = 0
    if end == -1:
        end = NN.shape[0] - 1
    debug_info = True
    
    if debug_info == True:
        print(NN[end])
        
        print(get_connections(NN, visited, 0))
    
    actual = start
    
    actual_path.append(start)
    
    solutions = []
        
    while visited.sum()!= NN.shape[0]:

        next_possibles = get_connections(NN, visited, actual)
        
        if (next_possibles == end).sum() > 0:
            if debug_info == True:
                print("***Es un posible")
                print(actual_path)
                print(np.where(next_possibles == end)[0][0])
            next_possibles = np.delete(next_possibles, np.where(next_possibles == end)[0][0])
            NN[actual, end] = 0
            sol = actual_path.copy()
            sol.append(end)
            solutions.append(sol)

            if debug_info == True:
                print("Soluciones: ", solutions)

        if debug_info == True:
            print("Desde: ", actual, " - Posibles siguientes: ", next_possibles.transpose(), " con los siguientes recorridos ", actual_path)
        
        # Si no hay más caminos desde el actual, unstackear
        if(next_possibles.shape[0] == 0):
            
            # if actual == start:
            if len(stack) == 0:
                break

            visited[actual] = 1
            actual = stack.pop()
            actual_path.pop()

        else:
                
            stack.append(actual)
            actual = next_possibles[0]
            actual_path.append(actual)

    if debug_info == True:
        print(solutions)

    return solutions

def min_cost(NN_w_costs, path, residual_G = None):
    # Falta contemplar cuando path = 0
    first_path = path[0]
    minimal = np.inf
    
    NN_w_costs_return = NN_w_costs.copy()
    
    if residual_G == None:
        residual_G = np.zeros(NN_w_costs.shape)
        
    residual_G_tmp = np.zeros(NN_w_costs.shape)
    
    for i in range(1, len(path)):
        value = NN_w_costs[path[i-1], path[i]]
        if value < minimal:
            minimal = value
        residual_G_tmp[path[i-1], path[i]] = 1
    residual_G_tmp = residual_G_tmp * minimal
    residual_G = residual_G_tmp + residual_G
    NN_w_costs_return = NN_w_costs_return - residual_G_tmp
    print("Minimal cost = ", minimal)
    return residual_G, NN_w_costs_return

def convert_cost_to_NxN(cost, idxs):
    NN = np.zeros((max(max(idxs)) +1, max(max(idxs)) + 1))
    for idx, i in enumerate(cost):
        NN[idxs[idx][0], idxs[idx][1]] = i
    return NN

def max_time_on_seq(times, paths, node_names = ""):
    max_times = []
    for path in paths:
        max_time = 0
        printable = ""

        if(len(node_names)>0):
            printable = printable + node_names[path[0]]

        for i in range(1, len(path)):
            max_time += times[path[i-1],path[i]]
            if(len(node_names)>0):
                printable = printable + " -> " + str(node_names[path[i]]) + " c(" + str(times[path[i-1],path[i]]) + ")"

        if(len(node_names)>0):
            print(printable)
        max_times.append(max_time)
    return max_times

def unroll_time_serie(capacities, times, all_paths, max_steps, node_names, max_costs, debug = False):
    max_steps = max_steps + 1
    final_matrix_nn = np.zeros((2 + times.shape[0]*(max_steps), 2 + times.shape[0]*(max_steps)), dtype = 'l')
    final_matrix_capacities = np.zeros((2 + times.shape[0]*(max_steps), 2 + times.shape[0]*(max_steps)))
    new_node_names = [node_names[0]]
    for j in range(max_steps):
        for i in node_names:
            new_node_names.append(i + str(j))
    new_node_names.append(node_names[-1])
            
    for idx, each_path in enumerate(all_paths):

        actual_time = 0 # Columna desde
        for j in range(1, len(each_path)):
            to_time = actual_time + times[each_path[j-1], each_path[j]]
            if debug == True:
                print(each_path[j-1], each_path[j])
                print(node_names[each_path[j-1]], actual_time, " -> ", node_names[each_path[j]], to_time)
                
            for t in range(0,max_steps):#max_steps - max_costs):
#                 t = 1
                idx_from = 1 + actual_time*times.shape[0] + t*times.shape[0] + each_path[j-1]
                idx_to = 1 + to_time*times.shape[0] + t*times.shape[0] + each_path[j]

                if idx_from >=final_matrix_nn.shape[0] or idx_to >= final_matrix_nn.shape[1]:
#                     print("Salteo")
#                     print(idx_from, idx_to)
                    continue
                elif debug == True:
                    print("Tiempos")
                    print(new_node_names[idx_from], new_node_names[idx_to])
                    
                final_matrix_nn[idx_from, idx_to] = 1
                final_matrix_capacities[idx_from, idx_to] += capacities[each_path[j-1], each_path[j]]
                
            actual_time = to_time
        if debug == True:
            print()
            
    # Uno s con s0..sn <- s y t0..tn -> t
    for i in range(0, max_steps):
        # Vinculo los s
        final_matrix_nn[0, 1 + i*times.shape[0]] = 1
        # Vinculo los t
        final_matrix_nn[(i+1)*times.shape[0], 1  + times.shape[0]*(max_steps)] = 1
        
        # Las capacidades máximas de s -> s0..sN son infinitas
        final_matrix_capacities[0, 1 + i*times.shape[0]] = None
        # Las capacidades máximas de t <- t0..tN son infinitas
        final_matrix_capacities[(i+1)*times.shape[0], 1  + times.shape[0]*(max_steps)] = None
        
    # Finalmente uno t -> s
    final_matrix_nn[1  + times.shape[0]*(max_steps), 0] = 1
    
    # Las capacidad máxima de t -> s es infinita
    final_matrix_capacities[1  + times.shape[0]*(max_steps), 0] = None
    
    return final_matrix_nn, final_matrix_capacities, new_node_names

def get_col_capacites(nn_max, idxs, nn_min = None):
    capacities_min_out = np.zeros(len(idxs))
    capacities_max_out = np.zeros(len(idxs))
    o = []
    k = 0
    for i,j in idxs:
        
        minval = 0
        maxval = nn_max[i, j]
        
        if nn_min is not None:
            capacities_min_out[k] = nn_min[i, j]
        if np.isnan(maxval):
            maxval = None

        o.append([minval, maxval])
        
        i = i + 1
        
    bounds = tuple([(o[arcs][0],o[arcs][1]) for arcs in range(len(o))])

    return bounds

def get_costs(nn, idxs):
    capacities_max_out = np.zeros(len(idxs))

    for i, j in enumerate(idxs):
        # print(nn[j] )
        capacities_max_out[i] = nn[j] 

    return capacities_max_out

def connect_nodes(NN, node_names, name_from, name_to, cost = 1, verbose = False):
    node_from = np.argwhere(node_names == name_from)
    node_to = np.argwhere(node_names == name_to)

    if node_from.shape[0] == 0 or node_to.shape[0] == 0:
        print("error")
        return
    
    node_from = node_from[0]
    node_to = node_to[0]
    NN[node_from, node_to] = cost

    if verbose == True:
        print("Set ", node_names[node_from], node_names[node_to], " - Cost: ", cost)

def set_value_on_node(matrix_to_fill, node_names, name_from, value, verbose = False):
    node_from = np.argwhere(node_names == name_from)
    if node_from.shape[0] == 0:
        print("error")
        return
    node_from = node_from[0]
    matrix_to_fill[node_from] = value
    if verbose == True:
        print("Set ", node_names[node_from], " - Value: ", value)

def get_new_cost(cost, t, lamb):
    c_new = np.zeros(cost.shape)
    return cost + lamb*t
