# -*- coding: utf-8 -*-
"""
ACO Demo Script - Generate visualization with charging station visits
Uses lower SOC to force charging station usage
"""
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import os

class Sol():
    def __init__(self):
        self.node_seq = None
        self.cost = None
        self.routes = None
        self.cost_list = None


class EVRP_ACO():
    def __init__(self, static, dynamic, distances, slope,
                 custom_num=10, charge_num=4,
                 Start_SOC=25,  # Lower SOC to force charging
                 velocity=50, max_load=4, t_limit=10,
                 alpha=3, beta=1, rho=0.1, epochs=200, ant_number=80):

        self.static = static
        self.dynamic = dynamic
        self.distances = distances
        self.slope = slope

        self.demands = np.array(dynamic[1]).flatten() * max_load
        self.num = static.shape[1]
        self.custom_num = custom_num
        self.charge_num = charge_num

        self.max_load = max_load
        self.Start_SOC = Start_SOC
        self.t_limit = t_limit
        self.velocity = velocity

        # EV parameters
        self.mc = 4100
        self.g = 9.81
        self.w = 1000
        self.Cd = 0.7
        self.A = 6.66
        self.Ad = 1.2041
        self.Cr = 0.01
        self.motor_d = 1.18
        self.motor_r = 0.85
        self.battery_d = 1.11
        self.battery_r = 0.93

        self.custom_time = 0.33
        self.charging_time = 1

        # ACO parameters
        self.alpha = alpha
        self.beta = beta
        self.ant_number = ant_number
        self.rho = rho
        self.epochs = epochs
        self.Q = 5
        self.pheromone = np.ones(self.distances.shape) * 100
        self.best_solution = None
        self.solution_list = []

    def Soc_Consume(self, i, j, load):
        power = (0.5 * self.Cd * self.A * self.Ad * (self.velocity / 3.6) ** 2 +
                 (load * self.w + self.mc) * self.g * (self.slope[i, j] + self.Cr))

        if power >= 0:
            return self.motor_d * self.battery_d * power * self.distances[i, j] / 3600
        else:
            return self.motor_r * self.battery_r * power * self.distances[i, j] / 3600

    def getsolution(self):
        solution_list = []
        local_best_sol = Sol()
        local_best_sol.cost = float('inf')

        for k in range(self.ant_number):
            node_seq = []
            open_node = np.ones(self.num)
            open_node[0:self.charge_num + 1] = 0
            node_seq = [int(random.randint(self.charge_num + 1, self.num - 1))]
            now_node = node_seq[-1]
            open_node[now_node] = 0

            while any(open_node):
                next_node = self.searchNextNode(now_node, open_node)
                node_seq.append(next_node)
                open_node[next_node] = 0
                now_node = next_node

            sol = Sol()
            sol.node_seq = node_seq
            sol.cost, sol.routes, sol.cost_list = self.split_routes(node_seq)
            solution_list.append(sol)

            if sol.cost < local_best_sol.cost:
                local_best_sol = copy.deepcopy(sol)

        self.solution_list = copy.deepcopy(solution_list)
        if local_best_sol.cost < self.best_solution.cost:
            self.best_solution = copy.deepcopy(local_best_sol)

    def searchNextNode(self, now_node, open_node):
        total_prob = 0.0
        next_node = None
        prob = np.zeros(len(open_node))

        for i in range(len(open_node)):
            if open_node[i]:
                eta = abs(self.distances[now_node, 0] + self.distances[0, i] - self.distances[now_node, i])
                pheromone = self.pheromone[now_node, i]
                prob[i] = ((eta ** self.alpha) * (pheromone ** self.beta))
                total_prob += prob[i]

        if total_prob == 0:
            for i in range(len(open_node)):
                if open_node[i] == 1:
                    next_node = i
        else:
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(len(open_node)):
                if open_node[i]:
                    temp_prob -= prob[i]
                    if temp_prob < 0.0:
                        next_node = i
                        break

        return next_node

    def split_routes(self, node_seq):
        node_seq = node_seq.copy()
        node_seq.insert(0, 0)

        vehicle_routes = []
        vehicle_soc_list = []
        route = []
        load = self.max_load
        time = 0
        soc = self.Start_SOC
        time_list = []
        soc_list = []

        for i in range(1, len(node_seq)):
            node_idx = int(node_seq[i])
            demand_i = float(self.demands[node_idx])
            check_load = (load >= demand_i)

            if check_load:
                check_nextnode_soc = (soc >= (self.Soc_Consume(node_seq[i - 1], node_seq[i], load) +
                                              self.Soc_Consume(node_seq[i], 0, load - self.demands[node_seq[i]])))

                if check_nextnode_soc:
                    check_nextnode_time = ((time + (self.distances[node_seq[i-1], node_seq[i]] / self.velocity) +
                                           self.custom_time + (self.distances[node_seq[i], 0] / self.velocity)) <= self.t_limit)

                    if check_nextnode_time:
                        route.append(node_seq[i])
                        soc_list.append([self.Soc_Consume(node_seq[i-1], node_seq[i], load), 0])
                        soc = soc - self.Soc_Consume(node_seq[i-1], node_seq[i], load)
                        load = load - self.demands[node_seq[i]]
                        time_list.append((self.distances[node_seq[i-1], node_seq[i]] / self.velocity) + self.custom_time)
                        time = time + (self.distances[node_seq[i-1], node_seq[i]] / self.velocity) + self.custom_time
                    else:
                        vehicle_routes.append(route)
                        soc_list.append([self.Soc_Consume(node_seq[i-1], 0, load), 0])
                        vehicle_soc_list.extend(soc_list)
                        route = [node_seq[i]]
                        soc_list = [[self.Soc_Consume(0, node_seq[i], self.max_load), 0]]
                        soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                        load = self.max_load - self.demands[node_seq[i]]
                        time_list = [(self.distances[0, node_seq[i]] / self.velocity) + self.custom_time]
                        time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                else:
                    # Need to visit charging station
                    SOC_station = [self.Soc_Consume(node_seq[i-1], j, load) for j in range(1, self.charge_num + 1)]
                    min_soc = min(SOC_station)
                    min_index = SOC_station.index(min_soc) + 1

                    if soc >= min_soc:
                        check_nextstation_time = ((time + (self.distances[node_seq[i-1], min_index] / self.velocity) +
                                                  self.charging_time + (self.distances[min_index, 0] / self.velocity)) <= self.t_limit)

                        if check_nextstation_time:
                            # Add charging station to route
                            route.append(min_index)
                            soc_list.append([self.Soc_Consume(node_seq[i - 1], min_index, load), soc - min_soc - self.Start_SOC])
                            soc = self.Start_SOC
                            time = time + (self.distances[node_seq[i-1], min_index] / self.velocity) + self.charging_time
                            time_list.append((self.distances[node_seq[i-1], min_index] / self.velocity) + self.charging_time)

                            check_nextnode_time = ((time + (self.distances[min_index, node_seq[i]] / self.velocity) +
                                                   self.custom_time + (self.distances[node_seq[i], 0] / self.velocity)) <= self.t_limit)

                            if check_nextnode_time:
                                route.append(node_seq[i])
                                soc = soc - self.Soc_Consume(min_index, node_seq[i], load)
                                soc_list.append([self.Soc_Consume(min_index, node_seq[i], load), 0])
                                time = time + (self.distances[min_index, node_seq[i]] / self.velocity) + self.custom_time
                                time_list.append((self.distances[min_index, node_seq[i]] / self.velocity) + self.custom_time)
                                load = load - self.demands[node_seq[i]]
                            else:
                                vehicle_routes.append(route)
                                soc_list.append([self.Soc_Consume(min_index, 0, load), 0])
                                vehicle_soc_list.extend(soc_list)
                                route = [node_seq[i]]
                                soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                                soc_list = [[self.Soc_Consume(0, node_seq[i], self.max_load), 0]]
                                time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                                time_list = [(self.distances[0, node_seq[i]] / self.velocity) + self.custom_time]
                                load = self.max_load - self.demands[node_seq[i]]
                        else:
                            vehicle_routes.append(route)
                            soc_list.append([self.Soc_Consume(node_seq[i - 1], 0, load), 0])
                            vehicle_soc_list.extend(soc_list)
                            route = [node_seq[i]]
                            soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                            soc_list = [[self.Soc_Consume(0, node_seq[i], self.max_load), 0]]
                            time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                            time_list.append((self.distances[0, node_seq[i]] / self.velocity) + self.custom_time)
                            load = self.max_load - self.demands[node_seq[i]]
                    else:
                        vehicle_routes.append(route)
                        soc_list.append([self.Soc_Consume(node_seq[i - 1], 0, load), 0])
                        vehicle_soc_list.extend(soc_list)
                        route = [node_seq[i]]
                        soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                        soc_list = [[self.Soc_Consume(0, node_seq[i], self.max_load), 0]]
                        time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                        time_list = [(self.distances[0, node_seq[i]] / self.velocity) + self.custom_time]
                        load = self.max_load - self.demands[node_seq[i]]
            else:
                vehicle_routes.append(route)
                soc_list.append([self.Soc_Consume(node_seq[i - 1], 0, load), 0])
                vehicle_soc_list.extend(soc_list)
                route = [node_seq[i]]
                soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                soc_list = [[self.Soc_Consume(0, node_seq[i], self.max_load), 0]]
                time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                time_list = [(self.distances[0, node_seq[i]] / self.velocity) + self.custom_time]
                load = self.max_load - self.demands[node_seq[i]]

        vehicle_routes.append(route)
        soc_list.append([self.Soc_Consume(node_seq[-1], 0, load), 0])
        vehicle_soc_list.extend(soc_list)
        vehicle_soc = np.array(vehicle_soc_list)[:, 0].sum()

        return vehicle_soc, vehicle_routes, vehicle_soc_list

    def update_pheromone(self):
        self.pheromone = (1 - self.rho) * self.pheromone
        for sol in self.solution_list:
            routes = sol.routes
            for route in routes:
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    self.pheromone[from_node, to_node] += self.Q / sol.cost

    def plot_graph(self, solution, save_path="aco_demo_with_charging.png"):
        routes = copy.deepcopy(solution.routes)

        fig, ax = plt.subplots(figsize=(12, 10))

        xc = self.static[0, :]
        yc = self.static[1, :]

        # Plot customers (blue)
        ax.scatter(xc[self.charge_num + 1:self.custom_num + self.charge_num + 1],
                   yc[self.charge_num + 1:self.custom_num + self.charge_num + 1],
                   c='blue', s=100, zorder=5, label='Customers')

        # Plot charging stations (green)
        ax.scatter(xc[1:self.charge_num + 1], yc[1:self.charge_num + 1],
                   c='green', s=150, marker='s', zorder=5, label='Charging Stations')

        # Plot depot (red)
        ax.scatter(xc[0], yc[0], c='red', s=200, marker='*', zorder=5, label='Depot')

        # Label nodes
        for i in range(self.charge_num + 1, self.custom_num + self.charge_num + 1):
            ax.annotate(f'C{i-self.charge_num}', (xc[i], yc[i] + 2), fontsize=8, ha='center')

        for i in range(1, self.charge_num + 1):
            ax.annotate(f'S{i}', (xc[i], yc[i] + 2), fontsize=9, ha='center', fontweight='bold', color='green')

        ax.annotate('Depot', (xc[0], yc[0] + 3), fontsize=10, ha='center', fontweight='bold', color='red')

        # Plot routes with different colors
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

        charging_visits = 0
        for k, route in enumerate(routes):
            route_copy = [0] + route + [0]
            color = colors[k % len(colors)]

            for i in range(1, len(route_copy)):
                start = route_copy[i - 1]
                end = route_copy[i]

                # Check if this is a charging station visit
                if 1 <= end <= self.charge_num:
                    charging_visits += 1
                    # Highlight path to charging station
                    ax.annotate('', xy=(xc[end], yc[end]), xytext=(xc[start], yc[start]),
                               arrowprops=dict(arrowstyle='->', color='green', lw=2.5, ls='--'))
                else:
                    ax.annotate('', xy=(xc[end], yc[end]), xytext=(xc[start], yc[start]),
                               arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(f'ACO Solution: {len(routes)} routes, {charging_visits} charging stops\n'
                     f'Total Energy: {solution.cost:.2f} kWh | SOC: {self.Start_SOC} kWh', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path}")
        print(f"Charging station visits: {charging_visits}")
        return charging_visits

    def run(self):
        sol = Sol()
        sol.cost = float('inf')
        self.best_solution = sol

        for ep in range(self.epochs):
            self.getsolution()
            self.update_pheromone()
            if (ep + 1) % 50 == 0:
                print(f"Epoch {ep+1}/{self.epochs}, Best cost: {self.best_solution.cost:.2f}")

        return self.best_solution


def generate_instance(custom_num=10, charge_num=4, grid_size=100):
    """Generate a random EVRP instance"""
    total_nodes = 1 + charge_num + custom_num  # depot + stations + customers

    # Random coordinates
    coords = np.random.rand(2, total_nodes) * grid_size
    # Put depot at center
    coords[:, 0] = [grid_size/2, grid_size/2]

    # Random demands (only for customers)
    demands = np.zeros(total_nodes)
    demands[charge_num + 1:] = np.random.rand(custom_num) * 0.5 + 0.1  # 0.1 to 0.6

    # Compute distance matrix
    distances = np.zeros((total_nodes, total_nodes))
    for i in range(total_nodes):
        for j in range(total_nodes):
            distances[i, j] = np.sqrt((coords[0, i] - coords[0, j])**2 +
                                      (coords[1, i] - coords[1, j])**2)

    # Random slopes (small values)
    slopes = (np.random.rand(total_nodes, total_nodes) - 0.5) * 0.1

    # Dynamic array: [time_windows, demands]
    dynamic = np.zeros((2, total_nodes))
    dynamic[1, :] = demands

    return coords, dynamic, distances, slopes


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("ACO Demo: Generating instance with charging station usage")
    print("=" * 60)

    # Generate instance
    custom_num = 10
    charge_num = 4
    static, dynamic, distances, slopes = generate_instance(custom_num, charge_num, grid_size=100)

    # Run ACO with LOW SOC to force charging
    print("\nRunning ACO with Start_SOC=25 kWh (low battery to force charging)...")
    aco = EVRP_ACO(
        static=static,
        dynamic=dynamic,
        distances=distances,
        slope=slopes,
        custom_num=custom_num,
        charge_num=charge_num,
        Start_SOC=25,      # LOW SOC - will need charging
        velocity=50,
        max_load=4,
        t_limit=10,
        alpha=3,
        beta=1,
        rho=0.1,
        epochs=200,
        ant_number=80
    )

    solution = aco.run()

    print(f"\nBest solution cost: {solution.cost:.2f} kWh")
    print(f"Routes: {solution.routes}")

    # Plot and save
    charging_visits = aco.plot_graph(solution, "aco_demo_with_charging.png")

    if charging_visits == 0:
        print("\nNo charging visits detected. Trying with even lower SOC...")
        aco2 = EVRP_ACO(
            static=static,
            dynamic=dynamic,
            distances=distances,
            slope=slopes,
            custom_num=custom_num,
            charge_num=charge_num,
            Start_SOC=15,      # VERY LOW SOC
            velocity=50,
            max_load=4,
            t_limit=10,
            alpha=3,
            beta=1,
            rho=0.1,
            epochs=200,
            ant_number=80
        )
        solution2 = aco2.run()
        aco2.plot_graph(solution2, "aco_demo_with_charging.png")

    print("\nDone!")
