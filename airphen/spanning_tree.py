#!/usr/bin/python3
from collections import defaultdict, namedtuple

Arc = namedtuple('Arc', ('tail', 'weight', 'head', 'data'))

# The third argument of gen_spanning_arborescence funtion specifies whether the result returned will be a Minimum spanning tree or a Maximum spanning tree.
def gen_spanning_arborescence(arcs, sink, opt):
    if opt == "max":
        arcs = [Arc(arc.tail, arc.weight * (-1), arc.head, arc.data) for arc in arcs]
    good_arcs = []
    quotient_map = {arc.tail: arc.tail for arc in arcs}
    quotient_map[sink] = sink
    while True:
        gen_arc_by_tail_rep = {}
        successor_rep = {}
        for arc in arcs:
            if arc.tail == sink:
                continue
            tail_rep = quotient_map[arc.tail]
            head_rep = quotient_map[arc.head]
            if tail_rep == head_rep:
                continue
            if tail_rep not in gen_arc_by_tail_rep or gen_arc_by_tail_rep[tail_rep].weight > arc.weight:
                gen_arc_by_tail_rep[tail_rep] = arc
                successor_rep[tail_rep] = head_rep
        cycle_reps = find_cycle(successor_rep, sink)
        if cycle_reps is None:
            good_arcs.extend(gen_arc_by_tail_rep.values())
            return spanning_arborescence(good_arcs, sink, opt)
        cycle_reps = kind_of_sort_by_weights(arcs, cycle_reps)
        good_arcs.extend(gen_arc_by_tail_rep[cycle_rep] for cycle_rep in cycle_reps)
        cycle_rep_set = kind_of_sort_by_weights(arcs,cycle_reps)
        cycle_rep = cycle_rep_set.pop(0)
        quotient_map = {node: cycle_rep if node_rep in cycle_rep_set else node_rep for node, node_rep in quotient_map.items()}

def kind_of_sort_by_weights(arcs, cycle):
    max = get_weight(arcs, cycle[0], cycle[1])
    index = 0
    for i in range(len(cycle)-1):
        if(get_weight(arcs, cycle[i], cycle[i+1]) > max):
            index = i
            max = get_weight(arcs, cycle[i], cycle[i+1])
    if get_weight(arcs,cycle[-1],cycle[0]) > max:
        index = -1
    cycle[0],cycle[index] = cycle[index],cycle[0]
    return cycle

def get_weight(arcs, tail, head):
    for arc in arcs:
        if(arc.tail == tail and arc.head == head):
            return arc.weight

def find_cycle(successor, sink):
    visited = {sink}
    for node in successor:
        cycle = []
        while node not in visited:
            visited.add(node)
            cycle.append(node)
            node = successor[node]
        if node in cycle:
            return cycle[cycle.index(node):]
    return None

def spanning_arborescence(arcs, sink, opt):
    arcs_by_head = defaultdict(list)
    for arc in arcs:
        if arc.tail == sink:
            continue
        arcs_by_head[arc.head].append(arc)
    solution_arc_by_tail = {}
    stack = arcs_by_head[sink]
    while stack:
        arc = stack.pop()
        if arc.tail in solution_arc_by_tail:
            continue
        if opt == "min":
            solution_arc_by_tail[arc.tail] = arc
        if opt == "max":
            solution_arc_by_tail[arc.tail] = Arc(arc.tail, arc.weight*(-1), arc.head, arc.data)
        stack.extend(arcs_by_head[arc.tail])
    return solution_arc_by_tail