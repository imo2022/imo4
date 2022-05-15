#%%

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random


def load_file(filename):
    return np.loadtxt(filename, delimiter=" ", usecols=(1, 2), dtype='int32', skiprows=6, comments='EOF')


def get_distances(points):
    return np.array([[distance.euclidean(points[x], points[y])
                      for x in range(len(points))] for y in range(len(points))])


def score_cycles(distances, remaining, cycle):
    scores = np.array([[distances[cycle[c - 1], r] + distances[cycle[c], r] -
                        distances[cycle[c - 1], cycle[c]] for c in range(len(cycle))] for r in remaining])
    return scores


def draw_path(points, cycle1, cycle2, save_file=None):
    cycle1.append(cycle1[0])
    cycle2.append(cycle2[0])

    c1 = np.array(points[cycle1, :])
    c2 = np.array(points[cycle2, :])

    plt.scatter(points[:, 0], points[:, 1], color='black')

    plt.plot(c1[:, 0], c1[:, 1], color='blue')
    plt.plot(c2[:, 0], c2[:, 1], color='red')
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()
    del cycle1[-1]
    del cycle2[-1]


def cycle_length(distances, cycle):
    c = cycle.copy()
    c.append(c[0])
    dist = sum(distances[c[i], c[i+1]] for i in range(len(c) - 1))
    return dist

def score(distances, cycles):
    return cycle_length(distances, cycles[0]) + cycle_length(distances, cycles[1])

def init_cycles(points, distances):
    cycle1, cycle2 = [], []
    start_point = np.random.randint(0, len(points))
    cycle1.append(start_point)

    start_point2 = np.argmax(distances[start_point, :])
    cycle2.append(start_point2)

    remaining = list(range(len(points)))
    remaining.remove(start_point)
    remaining.remove(start_point2)

    return cycle1, cycle2, remaining


def regret_method(points, distances, cycle1=None, cycle2=None, remaining=None):
    if cycle1 == None:
        cycle1, cycle2, remaining = init_cycles(points, distances)
        point = remaining[np.argmin(distances[cycle1[-1], remaining])]
        cycle1.append(point)
        remaining.remove(point)
        point = remaining[np.argmin(distances[cycle2[-1], remaining])]
        cycle2.append(point)
        remaining.remove(point)

    while len(remaining) > 0:
        for cycle in [cycle1, cycle2]:
            score = score_cycles(distances, remaining, cycle)
            regret = np.diff(np.partition(score, 1)[:, :2]).reshape(-1)
            weight = 1.7 * np.min(score, axis=1) - regret
            best_point = np.argmin(weight)
            best_insert = np.argmin(score[best_point])

            cycle.insert(best_insert, remaining[best_point])
            remaining.remove(remaining[best_point])
    return cycle1, cycle2


def random_solution(points, distances):
    cycle1, cycle2 = [], []
    remaining = list(range(len(points)))

    while len(remaining) > 0:
        for cycle in [cycle1, cycle2]:
            random_point = random.choice(remaining)
            cycle.append(random_point)
            remaining.remove(random_point)
    return cycle1, cycle2


def change_vertices(cycle1, cycle2, i, j, distances):
    l1 = len(cycle1)
    l2 = len(cycle2)
    c1 = cycle1.copy()
    c2 = cycle2.copy()
    inx, inx1, inx0 = c1[i], c1[(i - 1) % l1], c1[(i + 1) % l1]
    jnx, jnx1, jnx0 = c2[j], c2[(j - 1) % l2], c2[(j + 1) % l2]
    d1 = distances[inx][inx1] + distances[inx][inx0] + distances[jnx][jnx0] + distances[jnx][jnx1]
    d2 = distances[jnx][inx1] + distances[jnx][inx0] + distances[inx][jnx0] + distances[inx][jnx1]
    c1[i], c2[j] = c2[j], c1[i]
    return c1, c2, d2 - d1

def change_edges_inside_1(cycle, i, j, distance):
    if(abs(i-j) <= 2 or abs(i-j) > (len(cycle)-1)):
        return [], 0.0
    if i > j:
        i, j = j, i
    l1 = len(cycle)
    c1 = cycle.copy()
    inx, inx1 = c1[i], c1[(i + 1)%l1]
    jnx, jnx1 = c1[j], c1[(j + 1)%l1]
    d1 = distance[inx][inx1] + distance[jnx][jnx1]
    d2 = distance[inx][jnx] + distance[jnx1][inx1]
    c1[(i + 1)%l1 + 1:j] = c1[(i + 1)%l1 + 1:j][::-1]
    c1[(i + 1)%l1], c1[j] = c1[j], c1[(i + 1)%l1]
    return c1, d1 - d2


def change_edges_inside(cycle, i, j, distances):
    i1, i2, j1, j2 = cycle[i], cycle[(i+1)%len(cycle)], cycle[j], cycle[(j+1)%len(cycle)]
    return distances[i1, j1] + distances[i2, j2] - distances[i1, i2] - distances[j1, j2]


def get_steepest_edge(cycle, distances, cycle_id):
    cycle_copy = cycle.copy()
    moves = []
    while len(cycle_copy):
        a = np.random.choice(cycle_copy, size=1, replace=False)
        index = np.where(cycle_copy == a)[0][0]
        cycle_copy = np.delete(cycle_copy, index)
        for i in range(len(cycle_copy)):
            if i == index:
                continue
            d = change_edges_inside(cycle, index, i, distances)
            if d < -0.001:
                x1, y1, z1 = cycle[(index-1) % len(cycle)], cycle[index], cycle[(index + 1) % len(cycle)]
                x2, y2, z2 = cycle[(i-1) % len(cycle)], cycle[i], cycle[(i+1) % len(cycle)]
                moves.append((d, 'edge', cycle_id, x1, y1, z1, x2, y2, z2))
    return moves


def get_steepest_vertex(cycle1, cycle2, distances):
    moves = []
    cycle_copy = cycle1.copy()
    while len(cycle_copy):
        a = np.random.choice(cycle_copy, size=1, replace=False)
        index = np.where(cycle_copy == a)[0][0]
        cycle_copy = np.delete(cycle_copy, index)
        for i in range(len(cycle2)):
            c1, c2, d = change_vertices(cycle1, cycle2, index, i, distances)
            if d < -0.001:
                x1, y1, z1 = cycle1[(index-1) % len(cycle1)], cycle1[index], cycle1[(index + 1) % len(cycle1)]
                x2, y2, z2 = cycle2[(i-1) % len(cycle2)], cycle2[i], cycle2[(i+1) % len(cycle2)]
                moves.append((d, 'vertex', None, x1, y1, z1, x2, y2, z2))
    return moves


def get_moves(cycle1, cycle2, distances):
    moves = get_steepest_vertex(cycle1, cycle2, distances)
    moves += get_steepest_edge(cycle1, distances, 0)
    moves += get_steepest_edge(cycle2, distances, 1)
    return moves


def get_cycle_index(cycles, i):
    if i in cycles[0]:
        return 0, np.where(np.array(cycles[0]) == i)[0][0]
    else:
        return 1, np.where(np.array(cycles[1]) == i)[0][0]


def apply_move(move, cycle1, cycle2, distances):
    _, type_of_transformation, id, _, i, _, _, j, _ = move
    if type_of_transformation == 'vertex':
        c1, i_index = get_cycle_index([cycle1, cycle2], i)
        c2, j_index = get_cycle_index([cycle1, cycle2], j)
        cycle1, cycle2, _ = change_vertices([cycle1, cycle2][c1], [cycle1, cycle2][c2], i_index, j_index, distances)
    else:
        cycle = [cycle1, cycle2][id]
        i_index = np.where(np.array(cycle) == i)[0][0]
        j_index = np.where(np.array(cycle) == j)[0][0]
        i2 = (i_index+1) % len(cycle)
        d = (j_index - i2) % len(cycle)
        for k in range(abs(d)//2+1):
            a, b = (i2+k)%len(cycle), (i2+d-k)%len(cycle)
            cycle[a], cycle[b] = cycle[b], cycle[a]
    return cycle1, cycle2


def steepest_search(points, distances, cycle1=None, cycle2=None):
    if cycle1 == None:
        cycle1, cycle2 = random_solution(points, distances)
    while True:
        moves = get_moves(cycle1, cycle2, distances)
        if not moves:
            break
        move = min(moves, key=lambda move: move[0])
        cycle1, cycle2 = apply_move(move, cycle1, cycle2, distances)
    return cycle1, cycle2

#%%
