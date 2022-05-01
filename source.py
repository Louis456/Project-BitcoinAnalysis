from copy import deepcopy
import csv
import time
import numpy as np
import pandas as pd
from collections import deque

class Edge():
    def __init__(self, source: int, target: int, weight: int, timestamp: float):
        self.source = source
        self.target = target
        self.weight = weight
        self.timestamp = timestamp

    def __eq__(self, other):
        return self.source == other.source and self.target == other.target \
                and self.weight == other.weight and self.timestamp == other.timestamp

class Graph():
    def __init__(self):
        self.adj = {} # adjacency list
        self.adjDirect = {} # directed adjacency list
        self.adjDirectBiggestWeight = {} # directed adjacency list keeping only the biggest weights
        self.pointedBy = {} # inverse directed adjacency list
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

        if (source, target) in self.adjDirectBiggestWeight:
            if abs(edge.weight) > abs(self.adjDirectBiggestWeight[(source, target)]):
                self.adjDirectBiggestWeight[(source, target)] = abs(edge.weight)
        else:
            self.adjDirectBiggestWeight[(source, target)] = abs(edge.weight)

        self.edges.append(edge)
        self.nE += 1

        self.minTimeStamp = min(self.minTimeStamp, edge.timestamp)
        self.maxTimeStamp = max(self.minTimeStamp, edge.timestamp)

    def getSortedEdgesByTimestamp(self, reverse=False):
        return sorted(self.edges, key=lambda e: e.timestamp, reverse=reverse)

    def getSortedEdgesByWeight(self, reverse=False):
        return sorted(self.edges, key=lambda e: e.weight, reverse=reverse)

    def getTimestampsInfo(self):
        sortedEdges = self.getSortedEdgesByTimestamp()
        self.medianTimeStamp = sortedEdges[self.nE // 2].timestamp
        return (self.minTimeStamp, self.medianTimeStamp, self.maxTimeStamp)

    def importDataframe(self, dataframe):
        for index, row in dataframe.iterrows():
            self.addEdge(Edge(row['Source'], row['Target'], row['Weight'], row['Timestamp']))

"""
TASK 1
"""
def basic_properties(dataframe):
    graph = Graph()
    graph.importDataframe(dataframe)
    return (connected_components(graph)[0], bridges(graph), local_bridges(graph))

def connected_components(graph: Graph):
    count = 0
    marked = set()
    stack = []

    #For task 2.4
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

            #Append current component and check if it is the biggest for task 2.4
            components.append(curr_component)
            if len(curr_component) > biggest_len:
                biggest_index = count-1
                biggest_len = len(curr_component)

    biggest_component = components[biggest_index]
    return (count, biggest_component)

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
                    # add neighbors but not the target
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
                #If u and v have no neighbors in common then it's a local bridge
                if (not (set(graph.adj[v]) & set(graph.adj[u]))):
                    local_bridges += 1
    return local_bridges


"""
TASK 2
"""
def total_triadic_closures(dataframe):
    graph = Graph()
    graph.importDataframe(dataframe)
    nEdges = graph.nE
    median = nEdges // 2

    sortedEdges = graph.getSortedEdgesByTimestamp()
    timestamps = [sortedEdges[median].timestamp]
    visitedEdges = set()

    triads_closed = 0
    triads_closed_over_time = [0]

    graph2 = Graph()
    for i in range(median+1):
        #Add edges till median
        s = sortedEdges[i].source
        t = sortedEdges[i].target
        #Only add the first edge if multiple edges between same nodes to keep the oldest
        if (s, t) not in visitedEdges:
            visitedEdges.add((s, t))
            visitedEdges.add((t, s))
            graph2.addEdge(sortedEdges[i])

    #Add edges from median
    for i in range(median+1,nEdges,1):
        newEdge = sortedEdges[i]
        s = newEdge.source
        t = newEdge.target
        #only add oldest
        if (s, t) not in visitedEdges:
            graph2.addEdge(newEdge)
            visitedEdges.add((s, t))
            visitedEdges.add((t, s))
            #Add number of closures created
            triads_closed += len(set(graph2.adj[s]) & set(graph2.adj[t]))
            triads_closed_over_time.append(triads_closed)
            timestamps.append(newEdge.timestamp)

    return (timestamps, triads_closed_over_time, triads_closed)


"""
TASK 3
"""
def end_balanced_degree(dataframe):
    graph = Graph()
    graph.importDataframe(dataframe)
    nEdges = graph.nE
    median = nEdges // 2

    #sort edges in increasing timestamp order
    sortedEdges = graph.getSortedEdgesByTimestamp()
    median_timestamp = sortedEdges[median].timestamp

    balanced = 0
    weakly_balanced = 0
    nb_triangles = 0
    edges = {}

    balance_over_time = []
    timestamps = []
    max_balance = -1
    timestamp_at_max = -1

    graph2 = Graph()
    #Add edge one by one
    for edge in sortedEdges:
        graph2.addEdge(edge)
        s = edge.source
        t = edge.target
        #Get third nodes of every triangle formed with this edge
        third_nodes = set(graph2.adj[s]) & set(graph2.adj[t])
        #If already in the graph then update the values
        if (s,t) in edges:
            prevWeight = edges[(s, t)]
            if (prevWeight < 0 and edge.weight >= 0) or (prevWeight >= 0 and edge.weight < 0):
                #Remove from the count balanced and weakly balanced triangle with previous weight
                withdraw_balanced, withdraw_weakly = edge_balance(s, t, prevWeight, third_nodes, edges)
                balanced -= withdraw_balanced
                weakly_balanced -= withdraw_weakly
                #Add to the count balanced and weakly balanced triangle with new weight
                added_to_balanced, added_to_weakly = edge_balance(s, t, edge.weight, third_nodes, edges)
                balanced += added_to_balanced
                weakly_balanced += added_to_weakly
        #If not in then compute new triangles
        else:
            nb_triangles += len(third_nodes)
            added_to_balanced, added_to_weakly = edge_balance(s, t, edge.weight, third_nodes, edges)
            balanced += added_to_balanced
            weakly_balanced += added_to_weakly

        #update weights
        edges[(s,t)] = edge.weight
        edges[(t,s)] = edge.weight
        if edge.timestamp >= median_timestamp:
            computed_score = (balanced + ((2 / 3) * weakly_balanced)) / nb_triangles if nb_triangles > 0 else 0
            balance_over_time.append(computed_score)
            timestamps.append(edge.timestamp)
            if (computed_score) > max_balance:
                max_balance = computed_score
                timestamp_at_max = edge.timestamp
    end_balance = balance_over_time[-1]

    return (timestamps, balance_over_time, max_balance, timestamp_at_max, end_balance)


"""
input : Given an edge (source,target,weight) and third nodes (triangles)
returns : number of balanced triangles and weakly-balanced triangles
"""
def edge_balance(source, target, weight, third_nodes, edges):
    weight_edge_1 = weight
    added_to_balanced = 0
    added_to_weakly = 0
    #For each triangle, compute positive and negative edges
    for node in third_nodes:
        weight_edge_2 = edges[(source, node)]
        weight_edge_3 = edges[(target, node)]
        nb_positive = 0
        nb_negative = 0
        for weight in (weight_edge_1, weight_edge_2, weight_edge_3):
            if weight >= 0:
                nb_positive += 1
            else:
                nb_negative += 1
        if nb_positive == 3 or (nb_positive == 1 and nb_negative == 2):
            added_to_balanced += 1
        elif nb_negative == 3:
            added_to_weakly += 1
    return added_to_balanced, added_to_weakly


"""
TASK 4
"""
def distances(dataframe):
    graph = Graph()
    graph.importDataframe(dataframe)
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
    #For the plot
    distances = []
    nbPaths = []
    for key, value in pathsPerDist.items():
        if key > 0:
            distances.append(key)
            nbPaths.append(value)
    return (distances, nbPaths, pathsPerDist, longestPathLength, longestPathStart, longestPathEnd)


"""
TASK 5
"""
def pagerank(dataframe):
    graph = Graph()
    graph.importDataframe(dataframe)
    nbNodes = len(graph.adj)

    page_ranks = {}
    for v in graph.adj:
        page_ranks[v] = 1/nbNodes
    d = 0.85
    epsilon = 10e-10
    it = 0
    has_converged = False
    while not has_converged:
        it += 1
        old_page_ranks = deepcopy(page_ranks)
        #For each vertice p , we update the score
        for p in graph.adj:
            flow_amount = 0
            if p in graph.pointedBy:
                #For each vertice n pointing to p
                for n in graph.pointedBy[p]:
                    sum_outgoing_weights = 0
                    #compute amout flowing out of n to p
                    for out in graph.adjDirect[n]:
                        sum_outgoing_weights += graph.adjDirectBiggestWeight[(n, out)]
                    flow_amount += (old_page_ranks[n] * graph.adjDirectBiggestWeight[(n, p)]) / sum_outgoing_weights
            #Update weighted page rank score
            page_ranks[p] = (1 - d) + d*flow_amount
        sum_diff = 0
        #Check convergence
        for i in page_ranks:
            sum_diff += abs(old_page_ranks[i]-page_ranks[i])
        if sum_diff < epsilon:
            has_converged=True
        print("iteration of pagerank : ",it)

    max_val = 0
    max_index = 0
    for v in page_ranks:
        if page_ranks[v] > max_val:
            max_val = page_ranks[v]
            max_index = v

    return max_index, max_val
