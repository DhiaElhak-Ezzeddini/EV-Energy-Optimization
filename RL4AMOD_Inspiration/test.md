```mermaid
flowchart TD
    subgraph A[Graph Generation / Loading]
        A1[network_gen_wrapper.py]
        A2[NetworkDatabase.load_or_create_network()]
        A3[(Road Graph: nx.MultiDiGraph)]
        A1 --> A2 --> A3
    end

    subgraph B[Environment: EVRoutingEnv]
        B1[Init vehicle params<br/>battery, min SoC, etc.]
        B2[_calculate_energy_kwh(u,v)<br/>Edge energy model]
        B3[step(action):<br/>apply energy model,<br/>update SoC, reward, done]
        B4[get_reachable_actions_indices()]
        B1 --> B2 --> B3
        B3 --> B4
    end

    subgraph C[GraphParser]
        C1[Precompute edge_index / edge_attr<br/>(length, slope, congestion, road_quality)]
        C2[Define node features:<br/>is_current, is_target, current_soc]
        C3[parse_obs(state) -> PyG Data]
        C1 --> C2 --> C3
    end

    subgraph D[DQN Agent]
        D1[ReplayBuffer]
        D2[select_action(state):<br/>epsilon-greedy]
        D3[learn():<br/>optimize Q-network]
        D4[policy_net: GNN_QNetwork]
        D5[target_net: GNN_QNetwork (frozen)]
        D2 --> D4
        D3 --> D4
        D3 --> D5
        D1 -.stores transitions.-> D3
    end

    subgraph E[GNN_QNetwork Architecture]
        E1[Input: PyG Data<br/>x (nodes), edge_index, edge_attr]
        E2[GCNConv #1]
        E3[ReLU]
        E4[GCNConv #2]
        E5[Node Embeddings h]
        E6[For each reachable neighbor:<br/>Concat(h_current, h_neighbor, edge_attr)]
        E7[MLP (Linear→ReLU→Linear)]
        E8[Q-values for reachable neighbors]
        E1 --> E2 --> E3 --> E4 --> E5 --> E6 --> E7 --> E8
    end

    subgraph F[Training Loop (train_ev.py)]
        F1[Hydra Config (cfg)]
        F2[Initialize env, parser, agent]
        F3[for episode in range(N):]
        F4[Reset env → state_tuple]
        F5[Agent.select_action()]
        F6[Env.step(action)]
        F7[Agent.store_transition()]
        F8[Agent.learn()]
        F9[Log reward, energy, loss]
        F1 --> F2 --> F3
        F3 --> F4 --> F5 --> F6 --> F7 --> F8 --> F9
    end

    %% Data flow connections
    A3 --> B
    A3 --> C
    B --> F
    C --> D
    D --> F
    E --> D
```