import networkx as nx
import csv
from source import *
from graphplots import *

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

"""
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
"""
graphTest.addEdge(Edge(11, 12, -1, 10))
graphTest.addEdge(Edge(12, 13, -1, 10))
graphTest.addEdge(Edge(13, 14, -1, 10))
graphTest.addEdge(Edge(14, 11, -1, 10))
graphTest.addEdge(Edge(15, 16, -1, 10))
graphTest.addEdge(Edge(16, 17, -1, 10))
graphTest.addEdge(Edge(17, 15, -1, 10))
"""

"""
print("degree : ",balance_degree(graphTest)[2])

print("nb of cc : ",connected_components(graphTest)[0])
print("components : ",connected_components(graphTest)[1])
"""

print("number of Edges : ", graph.getCountEdges())
print("number of Vertices : ", graph.getCountVertices())

timestampInfo = graph.getTimestampsInfo()
print("min Timestamp : ", timestampInfo[0])
print("median Timestamp : ", timestampInfo[1])
print("max Timestamp : ", timestampInfo[2])

"""
    Task 1
"""
print("number of CC : ", connected_components(graph)[0])
print("NX number of CC : ", nx.number_connected_components(nxGraph))

#print("number of bridges : ",bridges(graph))
print("NX number of bridges : ", len(list(nx.bridges(nxGraph))))

print("number of local bridges : ", local_bridges(graph))
print("NX number of local bridges : ", len(list(nx.local_bridges(nxGraph, with_span=False))))


"""
    Task 2
"""
triad_res = triadic_closures(graph)
print("number or triad closed : ", triad_res[2])
name = "Accumulated number of triadic closures over time"
xlabel = "Timestamps"
ylabel = "Triadic closures"
out = "triadic_closures"
print_plot(triad_res[0], triad_res[1], name, xlabel, ylabel, out)


"""
    Task 3
"""
balance_degree_res = balance_degree(graph)
print("highest balance degree : ", balance_degree_res[2])
print("highest balance degree at the timestamp : ", balance_degree_res[3])
name = "Balance degree over time"
xlabel = "Timestamps"
ylabel = "Balance degree"
out = "balance_degree"
print_plot(balance_degree_res[0], balance_degree_res[1], name, xlabel, ylabel, out)


"""
    Task 4
"""
path_res = shortest_paths(graph)
print("distance and nb path :  ", path_res[2])
print("Longest path is", path_res[3], " from", path_res[4], " to", path_res[5])
name = "Number of shortest paths having a given distance"
xlabel = "Distance"
ylabel = "Number of shortest paths"
out = "shortest_paths"
print_plot(path_res[0], path_res[1], name, xlabel, ylabel, out)


"""
    Task 5
"""
pagerank_res = page_rank(graph)
print("page rank index :"+str(pagerank_res[0])+", value :"+str(pagerank_res[1]))
