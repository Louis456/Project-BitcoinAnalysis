import csv
import time
import numpy as np
from collections import deque

class Edge():
    def __init__(self, source, target, weight, timestamp):
        self.source = source
        self.target = target
        self.weight = weight
        self.timestamp = timestamp

    def __eq__(self, other):
        return self.source == other.source and self.target == other.target \
                and self.weight == other.weight and self.timestamp == other.timestamp

    def __lt__(self, other):
        return self.weight < other.weight

    def __le__(self, other):
        return self.weight <= other.weight

    def __gt__(self, other):
        return self.weight > other.weight

    def __ge__(self, other):
        return self.weight >= other.weight


class Graph():
    def __init__(self):
        self.adj = {}
        self.adjDirect = {}
        self.biggestWeight = {}
        self.pointedBy = {}
        self.edges = []
        self.nE = 0
        self.nV = 0
        self.minTimeStamp = float('inf')
        self.maxTimeStamp = float('-inf')
        self.medianTimeStamp = None

    def addEdge(self, edge: Edge):
        source = edge.source
        target = edge.target
        if source not in self.adj:
            self.adj[source] = [target]
            self.nV += 1
        elif target not in self.adj[source]:
            self.adj[source].append(target)
        if target not in self.adj:
            self.adj[target] = [source]
            self.nV += 1
        elif source not in self.adj[target]:
            self.adj[target].append(source)

        if source not in self.adjDirect:
            self.adjDirect[source] = [target]
        elif target not in self.adjDirect[source]:
            self.adjDirect[source].append(target)

        if target not in self.pointedBy:
            self.pointedBy[target] = [source]
        elif source not in self.pointedBy[target]:
            self.pointedBy[target].append(source)

        if (source, target) in self.biggestWeight:
            if abs(edge.weight) > abs(self.biggestWeight[(source, target)]):
                self.biggestWeight[(source, target)] = abs(edge.weight)
        else:
            self.biggestWeight[(source, target)] = edge.weight

        self.edges.append(edge)
        self.nE += 1

        self.minTimeStamp = min(self.minTimeStamp, edge.timestamp)
        self.maxTimeStamp = max(self.minTimeStamp, edge.timestamp)

    def removeEdge(self, edge: Edge):
        pass

    def getSortedEdgesByTimestamp(self, reverse=False):
        return sorted(self.edges, key=lambda e: e.timestamp, reverse=reverse)

    def getSortedEdgesByWeight(self, reverse=False):
        return sorted(self.edges, key=lambda e: e.weight, reverse=reverse)

    def getCountEdges(self):
        return self.nE

    def getCountVertices(self):
        return self.nV

    def getTimestampsInfo(self):
        sortedEdges = self.getSortedEdgesByTimestamp()
        self.medianTimeStamp = sortedEdges[self.nE // 2].timestamp
        return (self.minTimeStamp, self.medianTimeStamp, self.maxTimeStamp)

    def importCSV(self, filename):
        with open(filename) as csvFile:
            dataset = csv.reader(csvFile, delimiter=',')
            lineCount = 0
            for row in dataset:
                if lineCount > 0:
                    source = int(row[1])
                    target = int(row[2])
                    weight = int(row[3])
                    timestamp = float(row[4])
                    edge = Edge(source, target, weight, timestamp)
                    self.addEdge(edge)
                lineCount += 1

"""
TASK 1
"""
def connected_components(graph: Graph):
    """ O(v + e) """
    count = 0
    marked = set()
    stack = []
    components = []
    biggest_len = 0
    biggest_index = 0
    for v in graph.adj:
        if v not in marked:
            marked.add(v)
            count += 1
            curr_component = []
            # DFS
            stack.append(v)
            while len(stack) > 0:
                curr = stack.pop()
                curr_component.append(curr)
                for n in graph.adj[curr]:
                    if n not in marked:
                        marked.add(n)
                        stack.append(n)
            components.append(curr_component)
            if len(curr_component) > biggest_len:
                biggest_index = count-1
                biggest_len = len(curr_component)


    return (count, components[biggest_index])

def bridges(graph: Graph):
    count = 0
    bridgeMarked = set()
    stack = []
    for fr in graph.adj:
        if fr not in bridgeMarked:
            bridgeMarked.add(fr)
            for to in graph.adj[fr]:
                if to not in bridgeMarked:
                    # dfs to search for an alternative path
                    marked = set()
                    marked.add(fr)
                    found = False
                    # add neighbors without the to of the current tested edge
                    for n in graph.adj[fr]:
                        marked.add(n)
                        if n != to:
                            stack.append(n)
                    while len(stack) > 0:
                        curr = stack.pop()
                        for n in graph.adj[curr]:
                            if n == to:
                                found = True
                                stack = []
                                break
                            elif n not in marked:
                                marked.add(n)
                                stack.append(n)

                    if not found:
                        count += 1
    return count


def local_bridges(graph: Graph):
    local_bridges = 0
    marked = set()
    for v in graph.adj:
        marked.add(v)
        for u in graph.adj[v]:
            if u not in marked:
                if (not (set(graph.adj[v]) & set(graph.adj[u]))):
                    local_bridges += 1
    return local_bridges

"""
TASK 2
"""
def triadic_closures(graph):
    nEdges = graph.getCountEdges()
    median = nEdges // 2
    sortedEdges = graph.getSortedEdgesByTimestamp()

    triads_closed = 0
    triads_closed_over_time = [0]
    timestamps = [sortedEdges[median].timestamp]

    graph2 = Graph()
    for i in range(median+1):
        #ONLY ADD OLDEST EDGE
        s = sortedEdges[i].source
        t = sortedEdges[i].target
        if (s not in graph2.adj or t not in graph2.adj or (s not in graph2.adj[t] or t not in graph2.adj[s])):
            graph2.addEdge(sortedEdges[i])

    for i in range(median+1,nEdges,1):
        newEdge = sortedEdges[i]
        s = newEdge.source
        t = newEdge.target
        if (s not in graph2.adj or t not in graph2.adj or (s not in graph2.adj[t] or t not in graph2.adj[s])):
            graph2.addEdge(newEdge)
            triads_closed += len(set(graph2.adj[s]) & set(graph2.adj[t]))
            triads_closed_over_time.append(triads_closed)
            timestamps.append(newEdge.timestamp)

    return (timestamps, triads_closed_over_time, triads_closed)


"""
TASK 3
"""
def balance_degree(graph):
    nEdges = graph.getCountEdges()
    median = nEdges // 2

    reverseSortedEdges = graph.getSortedEdgesByTimestamp(reverse=True)
    sortedEdges = graph.getSortedEdgesByTimestamp()
    median_timestamp = reverseSortedEdges[median].timestamp

    balanced = 0
    weakly_balanced = 0
    nb_triangles = 0
    edges = {}
    graph2 = Graph()
    #ONLY ADD LATEST EDGE UP TO MEDIAN
    for i in range(median, nEdges):
        edge = reverseSortedEdges[i]
        s = edge.source
        t = edge.target
        #IF LATEST EDGE NOT ALREADY IN THE GRAPH
        if (s not in graph2.adj or t not in graph2.adj or (s not in graph2.adj[t] or t not in graph2.adj[s])):
            graph2.addEdge(edge)
            third_nodes = set(graph2.adj[s]) & set(graph2.adj[t])
            nb_triangles += len(third_nodes)
            weight_edge_1 = edge.weight
            added_to_balanced = 0
            added_to_weakly = 0
            for node in third_nodes:
                weight_edge_2 = edges[(s,node)][2]
                weight_edge_3 = edges[(t,node)][2]
                nb_positive = 0
                nb_negative = 0
                for weight in (weight_edge_1, weight_edge_2, weight_edge_3):
                    if weight >= 0:
                        nb_positive += 1
                    else:
                        nb_negative += 1
                if nb_positive == 3 or (nb_positive == 1 and nb_negative == 2):
                    added_to_balanced += 1
                    added_to_weakly += 1
                elif nb_negative == 3:
                    added_to_weakly += 1

            balanced += added_to_balanced
            weakly_balanced += added_to_weakly
            edges[(s,t)] = (added_to_balanced, added_to_weakly, edge.weight)
            edges[(t,s)] = (added_to_balanced, added_to_weakly, edge.weight)

    score_over_time = [(balanced + 2 / 3 * weakly_balanced) / nb_triangles] if nb_triangles > 0 else [0]
    timestamps = [median_timestamp]

    max_score = score_over_time[0]
    timestamp_at_max = timestamps[0]

    #ADD EDGES ONE BY ONE FROM MEDIAN AND UPDATE BALANCE DEGREE
    for i in range(median+1, nEdges,1):
        edge = sortedEdges[i]
        graph2.addEdge(edge)
        s = edge.source
        t = edge.target
        third_nodes = set(graph2.adj[s]) & set(graph2.adj[t])
        if (s,t) in edges:
            addedTo = edges[(s,t)]
            balanced -= addedTo[0]
            weakly_balanced -= addedTo[1]
        else:
            nb_triangles += len(third_nodes)
        weight_edge_1 = edge.weight
        added_to_balanced = 0
        added_to_weakly = 0
        for node in third_nodes:
            weight_edge_2 = edges[(s,node)][2]
            weight_edge_3 = edges[(t,node)][2]
            nb_positive = 0
            nb_negative = 0
            for weight in (weight_edge_1, weight_edge_2, weight_edge_3):
                if weight >= 0:
                    nb_positive += 1
                else:
                    nb_negative += 1
            if nb_positive == 3 or (nb_positive == 1 and nb_negative == 2):
                added_to_balanced += 1
                added_to_weakly += 1
            elif nb_negative == 3:
                added_to_weakly += 1

        balanced += added_to_balanced
        weakly_balanced += added_to_weakly
        edges[(s,t)] = (added_to_balanced, added_to_weakly, edge.weight)
        edges[(t,s)] = (added_to_balanced, added_to_weakly, edge.weight)
        computed_score = (balanced + 2 / 3 * weakly_balanced) / nb_triangles
        score_over_time.append(computed_score)
        timestamps.append(edge.timestamp)
        if (computed_score) > max_score:
            max_score = computed_score
            timestamp_at_max = edge.timestamp

    
    return (timestamps, score_over_time, max_score, timestamp_at_max)


"""
TASK 4
"""
def shortest_paths(graph):
    pathsPerDist = {} # ex: {0: 1412, 1:1325, 2:784, ...}
    longestPathStart = -1
    longestPathEnd = -1
    longestPathLength = -1

    biggestComp = connected_components(graph)[1]
    queue = deque() # linked-list

    for v in biggestComp:
        # BFS
        distTo = {}
        distTo[v] = 0
        queue.append(v)
        while len(queue) > 0:
            curr = queue.popleft()
            if distTo[curr] in pathsPerDist:
                pathsPerDist[distTo[curr]] += 1
            else:
                pathsPerDist[distTo[curr]] = 1
            if distTo[curr] > longestPathLength:
                longestPathLength = distTo[curr]
                longestPathEnd = curr
                longestPathStart = v
            if curr in graph.adjDirect:
                for n in graph.adjDirect[curr]:
                    if n not in distTo:
                        distTo[n] = distTo[curr] + 1
                        queue.append(n)

    pathsPerDist[0] = 0 # Remove paths of length 0 (the starting nodes)
    distances = pathsPerDist.keys()
    nbPaths = pathsPerDist.values()
    return (distances, nbPaths, pathsPerDist, longestPathLength, longestPathStart, longestPathEnd)


"""
TASK 5
"""
def page_rank(graph):
    nbNodes = len(graph.adj)

    page_ranks = np.full(nbNodes, 1/nbNodes)
    d = 0.85

    nEdges = graph.getCountEdges()
    """
    reverseSortedEdgesByWeight = graph.getSortedEdgesByWeight(reverse=True)
    graph2 = Graph()
    for i in range(nEdges):
        edge = reverseSortedEdgesByWeight[i]
        s = edge.source
        t = edge.target
        if s not in graph2.adjDirect or t not in graph2.adjDirect[s]:
            graph2.addEdge(edge)
    """

    has_converged = False
    while not has_converged:
        old_page_ranks = page_ranks
        for p in graph.adj:
            flow_amount = 0
            if p in graph.pointedBy:
                for n in graph.pointedBy[p]:
                    sum_outgoing_weights = 0
                    for out in graph.adjDirect[n]:
                        sum_outgoing_weights += graph.biggestWeight[(n, out)]
                    flow_amount += (page_ranks[n] * graph.biggestWeight[(n, p)]) / sum_outgoing_weights
            page_ranks[p] = (1 - d) + d*flow_amount

        if np.all(np.abs(old_page_ranks-page_ranks)) < 1e-8:
            has_converged = True

    return np.argmax(page_ranks), np.max(page_ranks)
