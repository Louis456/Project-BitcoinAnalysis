import csv

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
        self.edges = []
        self.sortedEdges = None
        self.reverseSortedEdges = None
        self.nE = 0
        self.nV = 0
        self.minTimeStamp = float('inf')
        self.maxTimeStamp = float('-inf')
        self.medianTimeStamp = None
        self.latestEdges = None
        self.hasChanged = False

    def addEdge(self, edge: Edge):
        self.hasChanged = True
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
        self.edges.append(edge)
        self.nE += 1

        self.minTimeStamp = min(self.minTimeStamp, edge.timestamp)
        self.maxTimeStamp = max(self.minTimeStamp, edge.timestamp)

    def removeEdge(self, edge: Edge):
        self.hasChanged = True
        pass

    def getSortedEdgesByTimestamp(self, reverse=False):
        if reverse == True:
            if self.reverseSortedEdges == None or self.hasChanged:
                self.reverseSortedEdges = sorted(self.edges, key=lambda e: e.timestamp, reverse=True)
            return self.reverseSortedEdges
        else:
            if self.sortedEdges == None or self.hasChanged:
                self.sortedEdges = sorted(self.edges, key=lambda e: e.timestamp, reverse=False)
            return self.sortedEdges

    def getSortedEdgesByWeight(self, reverse=False):
        pass

    def getLatestEdges(self):
        if self.latestEdges == None or self.hasChanged:
            self.hasChanged = False
            sortedEdges = self.getSortedEdgesByTimestamp(reverse=True)
            marked = set()
            self.latestEdges = {}
            for edge in self.edges:
                s = edge.source
                t = edge.target
                if (s,t) not in marked and (t,s) not in marked:
                    marked.add((s,t))
                    self.latestEdges[(s,t)] = edge
                    self.latestEdges[(t,s)] = edge
        return self.latestEdges

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
    for v in graph.adj:
        if v not in marked:
            count += 1
            # DFS
            stack.append(v)
            while len(stack) > 0:
                curr = stack.pop()
                for n in graph.adj[curr]:
                    if n not in marked:
                        marked.add(n)
                        stack.append(n)
    return count

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

    triads_closed = 0
    sortedEdges = graph.getSortedEdgesByTimestamp()

    nEdges = graph.getCountEdges()
    median = nEdges // 2
    graph2 = Graph()
    for i in range(median+1):
        #ONLY ADD LATEST EDGE
        s = sortedEdges[i].source
        t = sortedEdges[i].target
        if (s not in graph2.adj or t not in graph2.adj or (s not in graph2.adj[t] or t not in graph2.adj[s])):
            graph2.addEdge(sortedEdges[i])

    #print(graph2.adj)

    for i in range(median+1,nEdges,1):
        newEdge = sortedEdges[i]
        s = newEdge.source
        t = newEdge.target
        if (s not in graph2.adj or t not in graph2.adj or (s not in graph2.adj[t] or t not in graph2.adj[s])):
            graph2.addEdge(newEdge)
            triads_closed += len(set(graph2.adj[s]) & set(graph2.adj[t]))
        #print(triads_closed)
    return triads_closed


"""
TASK 3
"""
def balance_degree(graph):
    nEdges = graph.getCountEdges()
    median = nEdges // 2

    sortedEdges = graph.getSortedEdgesByTimestamp()
    latestEdges = graph.getLatestEdges()

    balanced = 0
    weakly_balanced = 0
    nb_triangles = 0

    graph2 = Graph()
    for i in range(median+1):
        #ONLY ADD LATEST EDGE
        s = sortedEdges[i].source
        t = sortedEdges[i].target
        if (s not in graph2.adj or t not in graph2.adj or (s not in graph2.adj[t] or t not in graph2.adj[s])):
            graph2.addEdge(sortedEdges[i])

    for i in range(median+1, nEdges):
        newEdge = sortedEdges[i]
        s = newEdge.source
        t = newEdge.target
        if (s not in graph2.adj or t not in graph2.adj or (s not in graph2.adj[t] or t not in graph2.adj[s])):
            graph2.addEdge(newEdge)
            third_nodes = set(graph2.adj[s]) & set(graph2.adj[t])
            nb_triangles += len(third_nodes)
            edge_1 = latestEdges[(s,t)]
            for node in third_nodes:
                edge_2 = latestEdges[(s,node)]
                edge_3 = latestEdges[(t,node)]
                nb_positive = 0
                nb_negative = 0
                for edge in (edge_1, edge_2, edge_3):
                    if edge.weight >= 0:
                        nb_positive += 1
                    else:
                        nb_negative += 1
                if nb_positive == 3 or (nb_positive == 1 and nb_negative == 2):
                    balanced += 1
                elif nb_negative == 3:
                    weakly_balanced += 1
                weakly_balanced += balanced

    return (balanced + 2 / 3 * weakly_balanced) / nb_triangles
