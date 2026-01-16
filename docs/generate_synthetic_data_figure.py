"""
Generate a visualization of synthetic training data for the DRL model.
This script creates a sample instance showing depot, charging stations, and customers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set seed for reproducibility
np.random.seed(42)

# Parameters (matching the training configuration)
num_customers = 10
num_charging_stations = 4

# Generate depot (center area)
depot = np.random.randint(25, 75, size=(2,))

# Generate depot charging station (near depot with small jitter)
jitter = np.random.randint(-3, 4, size=(2,))
if np.all(jitter == 0):
    jitter = np.array([2, -2])
depot_charging = np.clip(depot + jitter, 0, 100)

# Generate charging stations in 4 quadrants
stations = np.zeros((num_charging_stations, 2))
for i in range(num_charging_stations):
    if i % 4 == 0:  # Bottom-left quadrant
        stations[i] = [np.random.randint(0, 50), np.random.randint(0, 50)]
    elif i % 4 == 1:  # Top-left quadrant
        stations[i] = [np.random.randint(0, 50), np.random.randint(51, 101)]
    elif i % 4 == 2:  # Bottom-right quadrant
        stations[i] = [np.random.randint(51, 101), np.random.randint(0, 50)]
    else:  # Top-right quadrant
        stations[i] = [np.random.randint(51, 101), np.random.randint(51, 101)]

# Generate customer locations (random across the grid)
customers = np.random.randint(0, 101, size=(num_customers, 2))

# Generate random demands for customers (normalized)
demands = np.random.randint(1, 10, size=(num_customers,))

# Create the figure
fig, ax = plt.subplots(figsize=(10, 10))

# Plot customers with size proportional to demand
customer_scatter = ax.scatter(
    customers[:, 0], customers[:, 1],
    c='#2E86AB', s=demands * 40 + 80, marker='o',
    edgecolors='black', linewidths=1.5, zorder=3,
    label='Customers'
)

# Add customer labels
for i, (x, y) in enumerate(customers):
    ax.annotate(f'C{i+1}', (x, y), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

# Plot charging stations
station_scatter = ax.scatter(
    stations[:, 0], stations[:, 1],
    c='#28A745', s=200, marker='^',
    edgecolors='black', linewidths=1.5, zorder=4,
    label='Charging Stations'
)

# Add station labels
for i, (x, y) in enumerate(stations):
    ax.annotate(f'S{i+1}', (x, y), textcoords="offset points",
                xytext=(0, 14), ha='center', fontsize=9, fontweight='bold')

# Plot depot charging station
ax.scatter(
    depot_charging[0], depot_charging[1],
    c='#FFC107', s=250, marker='s',
    edgecolors='black', linewidths=2, zorder=5,
    label='Depot Charging'
)

# Plot depot
ax.scatter(
    depot[0], depot[1],
    c='#DC3545', s=350, marker='*',
    edgecolors='black', linewidths=2, zorder=6,
    label='Depot'
)
ax.annotate('Depot', (depot[0], depot[1]), textcoords="offset points",
            xytext=(0, 18), ha='center', fontsize=10, fontweight='bold')

# Draw quadrant boundaries (dashed lines)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Add quadrant labels
ax.text(25, 95, 'Quadrant II', ha='center', fontsize=9, color='gray', alpha=0.7)
ax.text(75, 95, 'Quadrant I', ha='center', fontsize=9, color='gray', alpha=0.7)
ax.text(25, 5, 'Quadrant III', ha='center', fontsize=9, color='gray', alpha=0.7)
ax.text(75, 5, 'Quadrant IV', ha='center', fontsize=9, color='gray', alpha=0.7)

# Set axis properties
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_title('Synthetic Training Data Instance\n(10 Customers, 4 Charging Stations)', fontsize=14, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Create legend
legend_elements = [
    plt.scatter([], [], c='#DC3545', s=200, marker='*', edgecolors='black', linewidths=1.5, label='Depot'),
    plt.scatter([], [], c='#FFC107', s=150, marker='s', edgecolors='black', linewidths=1.5, label='Depot Charging'),
    plt.scatter([], [], c='#28A745', s=120, marker='^', edgecolors='black', linewidths=1.5, label='Charging Station'),
    plt.scatter([], [], c='#2E86AB', s=120, marker='o', edgecolors='black', linewidths=1.5, label='Customer'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

# Add text box with instance info
info_text = f'Customers: {num_customers}\nCharging Stations: {num_charging_stations}\nGrid Size: 100×100'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()

# Save the figure
output_path = 'figures/synthetic_training_data.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to {output_path}")

plt.show()
