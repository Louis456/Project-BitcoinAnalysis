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

print("number of Edges : ", graph.getCountEdges())
print("number of Vertices : ", graph.getCountVertices())

timestampInfo = graph.getTimestampsInfo()
print("min Timestamp : ", timestampInfo[0])
print("median Timestamp : ", timestampInfo[1])
print("max Timestamp : ", timestampInfo[2])

print("number of CC : ",connected_components(graph))
print("NX number of CC : ",nx.number_connected_components(nxGraph))

#print("number of bridges : ",bridges(graph))
print("NX number of bridges : ",len(list(nx.bridges(nxGraph))))

print("number of local bridges : ",local_bridges(graph))
print("NX number of local bridges : ",len(list(nx.local_bridges(nxGraph, with_span=False))))

print("number or triad closed : ",triadic_closures(graph))
