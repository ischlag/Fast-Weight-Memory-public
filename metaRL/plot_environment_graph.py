def sample_adjacency_matrix(n_actions, n_islands, random_state):
    while True:
        A = np.zeros((n_actions, n_islands, n_islands))

        # every island has to be leavable by at least one means of transportation (and not not be ambiguous)
        for from_island in range(n_islands):
            to_island = random_state.choice([i for i in range(n_islands) if i != from_island])
            transport = random_state.randint(0, n_actions)
            A[transport, from_island, to_island] = 1

        # every island has to be reachable by one or more from-islands
        for to_island in range(n_islands):
            # only select from the islands that don't have any neighbours for a certain transportation
            transport_list, from_list = np.where(A.sum(2) == 0)
            # remove self from the selection
            options = np.asarray(list(filter(lambda x: x[0] != to_island, zip(from_list, transport_list))))
            indecies = np.arange(options.shape[0])
            chosen_idx = random_state.choice(indecies)
            from_island, transport = options[chosen_idx]
            A[transport, from_island, to_island] = 1

        # reject if they are not all connected
        Q = A.sum(0)
        Q[Q > 0] = 1
        for _ in range(n_islands):
            Q = np.matmul(Q,Q)
        if (Q == 0).sum() == 0:
            return A


n_states = 5
n_actions = 3
goal = 3
state = 2

import numpy as np
r = np.random.RandomState()
A = sample_adjacency_matrix(n_actions,n_states,r)

import networkx as nx
graph = nx.DiGraph()
graph.add_nodes_from(range(n_states))
edges = np.argwhere(A.max(axis=0))
edges = [tuple(e) + ({'action': tuple(np.argwhere(A[:, e[0], e[1]]).flatten())},) for e in edges]
graph.add_edges_from(edges)
pos = nx.spring_layout(graph)

# render
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
plt.close('all')
f, ax = plt.subplots()
canvas = FigureCanvasAgg(f)
node_colors = ['lime' if n == state else 'gold' if n == goal else '#03A9F4'
               for n in range(n_states)]
_ = nx.draw_networkx(graph, pos, connectionstyle='arc3, rad = 0.1', with_labels=True,
                 node_color=node_colors, ax=ax)
labels = dict([((u, v,), d['action']) for u, v, d in graph.edges(data=True)])
_ = nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, label_pos=0.5, ax=ax, font_size=11)
plt.savefig("graph_sample.png")

print(A)
A = A.sum(0)
A[A > 0] = 1
A

for i in range(n_states):
  A = np.matmul(A,A)

print((A == 0).sum())
print(A)
