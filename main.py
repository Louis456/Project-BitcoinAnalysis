import networkx as nx
import csv
from source import *

graph = Graph()
nxGraph = nx.Graph()

with open('Project dataset.csv') as csvFile:
    dataset = csv.reader(csvFile, delimiter=',')
    lineCount = 0
    for row in dataset:
        if lineCount > 0:
            source = row[1]
            target = row[2]
            weight = row[3]
            nxGraph.add_edge(source, target, weight=weight)
        lineCount += 1

graph.importCSV('Project dataset.csv')

graphTest = Graph()
graphTest.addEdge(Edge(1, 2, 1, 0))
graphTest.addEdge(Edge(1, 3, 1, 1))
graphTest.addEdge(Edge(2, 3, -1, 2))
graphTest.addEdge(Edge(2, 4, -1, 3))
graphTest.addEdge(Edge(3, 5, -1, 4))
graphTest.addEdge(Edge(3, 6, 1, 5))
graphTest.addEdge(Edge(4, 5, 1, 6))
graphTest.addEdge(Edge(4, 7, 1, 7))
graphTest.addEdge(Edge(6, 5, -1, 8))
graphTest.addEdge(Edge(7, 6, -1, 9))
graphTest.addEdge(Edge(7, 8, -1, 10))
graphTest.addEdge(Edge(7, 9, -1, 10))
graphTest.addEdge(Edge(8, 4, -1, 10))
graphTest.addEdge(Edge(9, 8, -1, 10))
graphTest.addEdge(Edge(9, 10, -1, 10))
print("graphTest adjDir : ",graphTest.adjDirect)
"""
graphTest.addEdge(Edge(11, 12, -1, 10))
graphTest.addEdge(Edge(12, 13, -1, 10))
graphTest.addEdge(Edge(13, 14, -1, 10))
graphTest.addEdge(Edge(14, 11, -1, 10))
graphTest.addEdge(Edge(15, 16, -1, 10))
graphTest.addEdge(Edge(16, 17, -1, 10))
graphTest.addEdge(Edge(17, 15, -1, 10))
"""


print("degree : ",balance_degree(graphTest)[2])

print("nb of cc : ",connected_components(graphTest)[0])
print("components : ",connected_components(graphTest)[1])

print("number of Edges : ", graph.getCountEdges())
print("number of Vertices : ", graph.getCountVertices())

timestampInfo = graph.getTimestampsInfo()
print("min Timestamp : ", timestampInfo[0])
print("median Timestamp : ", timestampInfo[1])
print("max Timestamp : ", timestampInfo[2])

print("number of CC : ", connected_components(graph)[0])
print("NX number of CC : ", nx.number_connected_components(nxGraph))

#print("number of bridges : ",bridges(graph))
print("NX number of bridges : ", len(list(nx.bridges(nxGraph))))

print("number of local bridges : ", local_bridges(graph))
print("NX number of local bridges : ", len(list(nx.local_bridges(nxGraph, with_span=False))))

print("number or triad closed : ", triadic_closures(graph)[2])

print("balance degree : ", balance_degree(graph)[2])

print("distance and nb path :  ", shortest_paths(graphTest)[2])

print("page rank index :"+str(page_rank(graph)[0])+", value :"+str(page_rank(graph)[1]))
