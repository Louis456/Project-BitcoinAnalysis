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
        self.edges.append(edge)
        self.nE += 1

        self.minTimeStamp = min(self.minTimeStamp, edge.timestamp)
        self.maxTimeStamp = max(self.minTimeStamp, edge.timestamp)

    def removeEdge(self, edge: Edge):
        pass

    def sortEdgesByTimestamp(self):
        self.edges = sorted(self.edges, key=lambda e: e.timestamp, reverse=False)

    def sortEdgesByWeight(self):
        self.edges = sorted(self.edges, key=lambda e: e.weight, reverse=False)

    def getCountEdges(self):
        return self.nE

    def getCountVertices(self):
        return self.nV

    def getTimestampsInfo(self):
        self.sortEdgesByTimestamp()
        self.medianTimeStamp = self.edges[self.nE // 2].timestamp
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
TASK1
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
TASK2
"""

def triadic_closures(graph):

    triads_closed = 0
    graph.sortEdgesByTimestamp()
    
    nEdges = graph.getCountEdges()
    median = nEdges // 2
    graph2 = Graph()
    for i in range(median):
        graph2.addEdge(graph.edges[i])
    for i in range(median+1,nEdges,1):
        newEdge = graph.edges[i]
        source = newEdge.source
        target = newEdge.target
        graph2.addEdge(newEdge)
        triads_closed += len(set(graph.adj[source]) & set(graph.adj[target]))
    return triads_closed




    # returns x, y with x=vector of timestamps and y=vector of accumulated closures since median
