import networkx as nx
import csv
from source import *
from graphplots import *
import pandas as pd

graph = Graph()
nxGraph = nx.Graph()

dataframe = pd.read_csv('Project dataset.csv')
for index, row in dataframe.iterrows():
    source = row[1]
    target = row[2]
    weight = row[3]
    nxGraph.add_edge(source, target, weight=weight)

graph.importDataframe(dataframe)
graphTest = Graph()

"""
graphTest.addEdge(Edge(0,1,4,1))
graphTest.addEdge(Edge(0,3,5,2))
graphTest.addEdge(Edge(1,2,2,3))
graphTest.addEdge(Edge(3,2,10,4))
graphTest.addEdge(Edge(3,4,7,5))
graphTest.addEdge(Edge(4,2,2,6))
graphTest.addEdge(Edge(4,0,1,7))
graphTest.addEdge(Edge(4,1,3,8))
pagerank_test_res = pagerank(graphTest)
print("page rank index :"+str(pagerank_test_res[0])+", value :"+str(pagerank_test_res[1]))
print("graphTest adjDirBiggestWeight : ",graphTest.biggestWeight)
print("graph adj :",graphTest.adj)
print("graph adjDirect :",graphTest.adjDirect)
print("graph pointedBy: ",graphTest.pointedBy)
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

print("number of Edges : ", graph.nE)
print("number of Vertices : ", graph.nV)


timestampInfo = graph.getTimestampsInfo()
print("min Timestamp : ", timestampInfo[0])
print("median Timestamp : ", timestampInfo[1])
print("max Timestamp : ", timestampInfo[2])


"""
    Task 1

number_connected_components, bridges, local_bridges = basic_properties(dataframe)
print("number of CC : ", number_connected_components)
print("NX number of CC : ", nx.number_connected_components(nxGraph))

print("number of bridges : ",bridges)
print("NX number of bridges : ", len(list(nx.bridges(nxGraph))))

print("number of local bridges : ", local_bridges)
print("NX number of local bridges : ", len(list(nx.local_bridges(nxGraph, with_span=False))))

"""
"""
    Task 2
"""
triad_res = total_triadic_closures(dataframe)
print("number or triad closed : ", triad_res[2])
name = "Accumulated number of triadic closures over time"
xlabel = "Timestamps"
ylabel = "Triadic closures"
out = "triadic_closures"
print_plot(triad_res[0], triad_res[1], name, xlabel, ylabel, out)


"""
    Task 3
"""
balance_degree_res = end_balanced_degree(dataframe)
print("highest balance degree : ", balance_degree_res[2])
print("highest balance degree at the timestamp : ", balance_degree_res[3])
print("end score balance :", balance_degree_res[4])
name = "Balance degree over time"
xlabel = "Timestamps"
ylabel = "Balance degree"
out = "balance_degree"
print_plot(balance_degree_res[0], balance_degree_res[1], name, xlabel, ylabel, out)


"""
    Task 4
"""
path_res = distances(dataframe)
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
pagerank_res = pagerank(dataframe)
print("page rank index :"+str(pagerank_res[0])+", value :"+str(pagerank_res[1]))
