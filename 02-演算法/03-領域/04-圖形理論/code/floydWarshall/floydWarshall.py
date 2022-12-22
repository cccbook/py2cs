# https://github.com/TheAlgorithms/Python/blob/master/graphs/graphs_floyd_warshall.py
# floyd_warshall.py
"""
    The problem is to find the shortest distance between all pairs of vertices in a
    weighted directed graph that can have negative edge weights.
"""

from math import inf

def _print_dist(dist, v):
    print("\nThe shortest path matrix using Floyd Warshall algorithm\n")
    for i in range(v):
        for j in range(v):
            if dist[i][j] != float("inf"):
                print(int(dist[i][j]), end="\t")
            else:
                print("INF", end="\t")
        print()


def floyd_warshall(graph, v):
    """
    :param graph: 2D array calculated from weight[edge[i, j]]
    :type graph: List[List[float]]
    :param v: number of vertices
    :type v: int
    :return: shortest distance between all vertex pairs
    distance[u][v] will contain the shortest distance from vertex u to v.
    1. For all edges from v to n, distance[i][j] = weight(edge(i, j)).
    3. The algorithm then performs distance[i][j] = min(distance[i][j], distance[i][k] +
        distance[k][j]) for each possible pair i, j of vertices.
    4. The above is repeated for each vertex k in the graph.
    5. Whenever distance[i][j] is given a new minimum value, next vertex[i][j] is
        updated to the next vertex[i][k].
    """

    dist = [[float("inf") for _ in range(v)] for _ in range(v)]

    for i in range(v):
        for j in range(v):
            dist[i][j] = graph[i][j]

            # check vertex k against all other vertices (i, j)
    for k in range(v):
        # looping through rows of graph array
        for i in range(v):
            # looping through columns of graph array
            for j in range(v):
                if (
                    dist[i][k] != float("inf")
                    and dist[k][j] != float("inf")
                    and dist[i][k] + dist[k][j] < dist[i][j]
                ):
                    dist[i][j] = dist[i][k] + dist[k][j]

    _print_dist(dist, v)
    return dist, v

# https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm#Example
if __name__ == "__main__":
    v = 4
    graph = [
        [0.0, inf, -2.0, inf], 
        [4.0, 0.0,  3.0, inf], 
        [inf, inf,  0.0, 2.0],
        [inf, -1.0, inf, 0.0]]

    floyd_warshall(graph, v)