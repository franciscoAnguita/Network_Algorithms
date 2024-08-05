%matplotlib notebook

from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

G = nx.Graph()

G.add_edge("a", "b", weight=0.6)
G.add_edge("a", "c", weight=0.2)
G.add_edge("c", "d", weight=0.1)
G.add_edge("c", "e", weight=0.7)
G.add_edge("c", "f", weight=0.9)
G.add_edge("a", "d", weight=0.3)

init_pos = nx.spring_layout(G, seed=7)

def movedge(pos, p1, p2, distance):
    nodes = p1, p2
    p1, p2 = itemgetter(p1, p2)(pos)
    curr_distance = np.linalg.norm(p2 - p1)
    scale = distance / curr_distance
    new_p1 = p1 + (p2 - p1) * scale
    new_p2 = p2 + (p1 - p2) * scale
    return {**pos, **dict(zip(nodes, (new_p1, new_p2)))}

def drawg(g, curr_pos, ax):
    ax.set(xlim=(-3, 3), ylim=(-1, 1))
    # NODES
    nx.draw_networkx_nodes(
        g, curr_pos, ax=ax,
        node_color="tab:red",
        edgecolors="tab:gray",
        node_size=400, alpha=0.9,
    )
    nx.draw_networkx_labels(
        g, curr_pos, ax=ax,
        font_size=22, font_color="whitesmoke",
    )
    # EDGES
    nx.draw_networkx_edges(
        g, curr_pos, ax=ax,
        width=4, alpha=0.7,
        edge_color="tab:blue", style="dashed",
    )
    nx.draw_networkx_edge_labels(
        g, curr_pos, ax=ax,
        font_size=10, font_weight="bold",
        edge_labels=nx.get_edge_attributes(g, "weight"),
    )

    
fig, ax = plt.subplots(figsize=(10, 6))

moves = ( # some random moves
    {"p1": "a", "p2": "b", "distance": 0.57},
    {"p1": "c", "p2": "d", "distance": 0.12},
    {"p1": "c", "p2": "f", "distance": 0.88},
    {"p1": "c", "p2": "d", "distance": 0.07},
    {"p1": "c", "p2": "e", "distance": 0.68},
    {"p1": "a", "p2": "b", "distance": 0.59},
)

N = 20
frames = {}
H = G.copy(); new_pos = init_pos.copy()
for move, rng in zip(moves, np.arange(1, len(moves) * N + 1).reshape(-1, N)):
    p1, p2 = move["p1"], move["p2"]; init = H[p1][p2]["weight"]
    for i, j in zip(rng, np.linspace(init, move["distance"], N).round(2)):
        H[p1][p2]["weight"] = j; move["distance"] = j-init
        new_pos.update(movedge(**move, pos=new_pos))
        frames[i] = {"g": H.copy(), "curr_pos": movedge(**move, pos=new_pos)}


def animate(i):
    ax.clear()
    drawg(**frames[i + 1], ax=ax)


ani = FuncAnimation(
    fig, animate, frames=N * len(moves), interval=150, repeat=True
)

writer = PillowWriter(
    fps=15, metadata=dict(artist="Me"), bitrate=1800,
)
ani.save("output.gif", writer=writer)