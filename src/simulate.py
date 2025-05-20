import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import networkx as nx
import plotly.express as px
from tqdm import tqdm

from brain import Brain

# Parameters
NUM_NEURONS = 100
TIME_STEPS = 500
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

brain = Brain(NUM_NEURONS)
firing_data = []

print("Starting simulation...")
for t in tqdm(range(TIME_STEPS)):
    firings = brain.step(t)
    firing_data.extend(firings)

# Save firing data
df = pd.DataFrame(firing_data, columns=["time_step", "neuron_id"])
csv_path = os.path.join(OUTPUT_DIR, "firing_data.csv")
df.to_csv(csv_path, index=False)
print(f"Firing data saved to {csv_path}")

# Raster Plot
plt.figure(figsize=(14, 6))
plt.eventplot([df[df["neuron_id"] == n]["time_step"] for n in range(NUM_NEURONS)],
              orientation='horizontal', colors='black', linelengths=0.7)
plt.title("Spike Raster Plot")
plt.xlabel("Time Step")
plt.ylabel("Neuron ID")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "firing_raster.png"))
plt.show()

# Activity over time plot
plt.figure(figsize=(12, 4))
counts = df.groupby("time_step").size()
plt.plot(counts.index, counts.values, color="blue")
plt.title("Total Neuron Activity Over Time")
plt.xlabel("Time Step")
plt.ylabel("Number of Neurons Fired")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "activity_over_time.png"))
plt.show()

# Synaptic weight heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(brain.weights, cmap="coolwarm", center=0, square=True)
plt.title("Synaptic Weight Matrix")
plt.xlabel("Post-Synaptic Neuron")
plt.ylabel("Pre-Synaptic Neuron")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "weight_heatmap.png"))
plt.show()

# Neural Network Graph (networkx)
G = nx.DiGraph()
threshold = 0.3  # plot threshold for visibility

for i in range(NUM_NEURONS):
    for j in range(NUM_NEURONS):
        w = brain.weights[i, j]
        if abs(w) > threshold:
            G.add_edge(i, j, weight=w)

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)
edges = G.edges(data=True)
colors = ['green' if d['weight'] > 0 else 'red' for (_, _, d) in edges]
weights = [abs(d['weight']) * 2 for (_, _, d) in edges]  # scale for visibility
nx.draw(G, pos, edges=edges, edge_color=colors, width=weights,
        with_labels=True, node_size=150, font_size=8)
plt.title("Neural Network Connectivity Graph")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "network_graph.png"))
plt.show()

# PCA on firing patterns
firing_matrix = np.zeros((NUM_NEURONS, TIME_STEPS))
for time_step, neuron_id in firing_data:
    firing_matrix[neuron_id, time_step] = 1

pca = PCA(n_components=2)
firing_pca = pca.fit_transform(firing_matrix)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(firing_pca[:, 0], firing_pca[:, 1], c=range(NUM_NEURONS), cmap='plasma', s=30)
plt.colorbar(scatter, label="Neuron ID")
plt.title("PCA of Neuron Firing Patterns")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_firing_patterns.png"))
plt.show()

# Interactive plotly plot
fig = px.scatter(df, x="time_step", y="neuron_id",
                 title="Interactive Neuron Firing Plot",
                 labels={"time_step": "Time Step", "neuron_id": "Neuron ID"},
                 template="plotly_dark", height=600)
fig.update_traces(marker=dict(size=6, opacity=0.7))
fig.write_html(os.path.join(OUTPUT_DIR, "interactive_firing_plot.html"))
fig.show()
