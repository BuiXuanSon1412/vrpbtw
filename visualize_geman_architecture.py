#!/usr/bin/env python3
"""
Visualization script for GEMAN architecture.
Generates Graphviz diagrams and ASCII art representations.

Usage:
    python3 visualize_geman_architecture.py [--format png|svg|pdf]
"""

import argparse
import sys
from pathlib import Path


def generate_graphviz_diagram() -> str:
    """Generate a Graphviz diagram of the GEMAN architecture."""
    return '''digraph GEMANArchitecture {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
    graph [rankdir=TB, bgcolor=white];
    edge [fontsize=9];

    // Color palette
    node [color="#E8F4F8"];

    // ========== INPUT LAYER ==========
    subgraph cluster_0 {
        label="INPUT LAYER";
        style=filled;
        color="#F0F0F0";
        fontsize=12;
        fontweight=bold;

        nf [label="node_features\\n(B, N+1, 5)", color="#D4E6F1", fontsize=10];
        vf [label="vehicle_features\\n(B, 2K, 5)", color="#D4E6F1", fontsize=10];
        te_idx [label="truck_edge_index\\n(2, E)", color="#D4E6F1", fontsize=9];
        te_attr [label="truck_edge_attr\\n(E, 2)", color="#D4E6F1", fontsize=9];
        de_idx [label="drone_edge_index\\n(2, E)", color="#D4E6F1", fontsize=9];
        de_attr [label="drone_edge_attr\\n(E, 2)", color="#D4E6F1", fontsize=9];
    }

    // ========== ENCODER LAYER ==========
    subgraph cluster_1 {
        label="ENCODER STAGE (Parallel)";
        style=filled;
        color="#F0F0F0";
        fontsize=12;
        fontweight=bold;

        node_enc [label="NodeEncoder\\n- Linear(5→D)\\n- 3× Att+FFN layers\\n- Adaptive pooling\\n\\nOutputs:\\n• Z_node (B,N+1,D)\\n• g_node (B,D)",
                  color="#FCF3CF", fontsize=10, width=2.5];

        veh_enc [label="VehicleEncoder\\n- Linear(5→D)\\n- 3× Att+FFN layers\\n- Adaptive pooling\\n\\nOutputs:\\n• Z_veh (B,2K,D)\\n• g_veh (B,D)",
                 color="#FCF3CF", fontsize=10, width=2.5];

        graph_enc [label="GraphEncoder (MRNN)\\n- Linear(4→D)\\n- 3× Multi-Relational layers\\n  - Truck relations\\n  - Drone relations\\n- Adaptive pooling\\n\\nOutputs:\\n• Z_graph (B,N+1,D)\\n• g_graph (B,D)",
                   color="#FCF3CF", fontsize=10, width=3];
    }

    // ========== DECODER LAYER ==========
    subgraph cluster_2 {
        label="DECODER STAGE (Bilevel)";
        style=filled;
        color="#F0F0F0";
        fontsize=12;
        fontweight=bold;

        node_dec [label="NodeDecoder\\n- ctx = Linear(3D→hidden)\\n- Q = Linear(hidden→D)\\n- Scores = softmax(Q·K^T/√D)\\n- Logits = clip·tanh(scores)\\n\\nOutput:\\n• node_logits (B,N+1)",
                  color="#D5F4E6", fontsize=10, width=2.5];

        veh_dec [label="VehicleDecoder\\n- ctx = Linear(3D→hidden)\\n- Q = Linear(hidden→D)\\n- Scores = softmax(Q·K^T/√D)\\n- Logits = clip·tanh(scores)\\n\\nOutput:\\n• veh_logits (B,2K)",
                 color="#D5F4E6", fontsize=10, width=2.5];

        joint [label="Joint Action Logits\\n\\nflat[b,i,j] = \\nnode_logits[b,i] + veh_logits[b,j]\\n\\nReshaped to (B, (N+1)×2K)\\nApply action_mask\\n\\nOutput:\\n• flat_logits (B, N·K×2)",
               color="#D5F4E6", fontsize=10, width=2.5];
    }

    // ========== VALUE HEAD ==========
    subgraph cluster_3 {
        label="VALUE HEAD";
        style=filled;
        color="#F0F0F0";
        fontsize=12;
        fontweight=bold;

        value_head [label="Value Head (MLP)\\n- Input: cat[g_node, g_veh, g_graph] (B, 3D)\\n- Linear(3D → h₀=128) + ReLU + Dropout\\n- Linear(h₀ → h₁=128) + ReLU + Dropout\\n- Linear(h₁ → 1)\\n\\nOutput:\\n• value (B,)",
                    color="#FADBD8", fontsize=10, width=2.8];
    }

    // ========== OUTPUT LAYER ==========
    subgraph cluster_4 {
        label="OUTPUT";
        style=filled;
        color="#F0F0F0";
        fontsize=12;
        fontweight=bold;

        policy_out [label="POLICY\\n\\nCategorical(logits=flat_logits)\\n- Sample actions: a ~ π(·|s)\\n- Log probabilities: log π(a|s)\\n- Entropy: H(π(·|s))",
                    color="#E8DAEF", fontsize=10, width=2.3];

        value_out [label="STATE VALUE\\n\\nV(s) scalar estimate\\nUsed for TD error,\\nadvantage computation,\\nbootstrap return",
                   color="#E8DAEF", fontsize=10, width=2.3];
    }

    // ========== EDGES: Input to Encoders ==========
    nf -> node_enc [label="5 features", fontsize=8];
    vf -> veh_enc [label="5 features", fontsize=8];

    nf -> graph_enc [label="Extract [x,y,tw_o,tw_c]→4D", fontsize=8, style=dashed];
    te_idx -> graph_enc [fontsize=8];
    te_attr -> graph_enc [fontsize=8];
    de_idx -> graph_enc [fontsize=8];
    de_attr -> graph_enc [fontsize=8];

    // ========== EDGES: Encoders to Decoders ==========
    node_enc -> node_dec [label="Z_node (B,N+1,D)\\ng_node (B,D)", fontsize=8];
    veh_enc -> veh_dec [label="Z_veh (B,2K,D)\\ng_veh (B,D)", fontsize=8];

    veh_enc -> node_dec [label="mean(Z_veh) cross-attn", fontsize=8, style=dashed, color=blue];
    node_enc -> veh_dec [label="mean(Z_node) cross-attn", fontsize=8, style=dashed, color=blue];

    graph_enc -> node_dec [label="g_graph", fontsize=8, color=red];
    graph_enc -> veh_dec [label="g_graph", fontsize=8, color=red];

    node_enc -> value_head [label="g_node", fontsize=8];
    veh_enc -> value_head [label="g_veh", fontsize=8];
    graph_enc -> value_head [label="g_graph", fontsize=8];

    // ========== EDGES: Decoders to Joint ==========
    node_dec -> joint [label="node_logits (B,N+1)", fontsize=8];
    veh_dec -> joint [label="veh_logits (B,2K)", fontsize=8];

    // ========== EDGES: Joint & Value to Output ==========
    joint -> policy_out [label="flat_logits", fontsize=8];
    value_head -> value_out [label="value", fontsize=8];

    // ========== LEGEND ==========
    subgraph cluster_legend {
        label="NOTATION";
        style=filled;
        color="#FFFACD";
        fontsize=11;

        l0 [label="D = embedding_dim (default 128)", shape=plaintext, fontsize=9];
        l1 [label="B = batch size", shape=plaintext, fontsize=9];
        l2 [label="N = number of nodes", shape=plaintext, fontsize=9];
        l3 [label="K = number of vehicles (per type)", shape=plaintext, fontsize=9];
        l4 [label="E = number of edges", shape=plaintext, fontsize=9];
        l5 [label="Solid lines = data flow, Dashed = attention/cross-modal", shape=plaintext, fontsize=9];
    }
}'''


def generate_simplified_ascii() -> str:
    """Generate a simplified ASCII architecture diagram."""
    return '''
╔════════════════════════════════════════════════════════════════════════════╗
║                    GEMAN ARCHITECTURE DATA FLOW                            ║
╚════════════════════════════════════════════════════════════════════════════╝

INPUT
─────
    node_features (B, N+1, 5)
    vehicle_features (B, 2K, 5)
    truck_edge_index, truck_edge_attr
    drone_edge_index, drone_edge_attr
           │
           │
           ├─────────┬──────────────┬────────────┐
           │         │              │            │
           ▼         ▼              ▼            ▼

    ╔═════════╗  ╔═════════╗  ╔═════════════╗
    │ Node    │  │Vehicle  │  │ Graph       │
    │Encoder  │  │Encoder  │  │ Encoder     │
    │         │  │         │  │ (MRNN)      │
    │ Linear  │  │ Linear  │  │             │
    │ 3×Attn  │  │ 3×Attn  │  │ 3×Multi-Rel │
    │ Pool    │  │ Pool    │  │ Layers      │
    │         │  │         │  │ Pool        │
    ╚────┬────╝  ╚────┬────╝  ╚────┬────────╝
         │            │            │
    Z_node(B,N,D) Z_veh(B,K,D) Z_graph(B,N,D)
    g_node(B,D)   g_veh(B,D)   g_graph(B,D)
         │            │            │
         └────────┬───┴────────────┘
                  │
        Cross-attention via global contexts
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
    ╔═════════════╗   ╔═════════════╗
    │ Node        │   │ Vehicle     │
    │ Decoder     │   │ Decoder     │
    │             │   │             │
    │ Context:    │   │ Context:    │
    │ Linear(3D)  │   │ Linear(3D)  │
    │ Q·K^T/√D    │   │ Q·K^T/√D    │
    │ Tanh clip   │   │ Tanh clip   │
    ╚────┬────────╝   ╚────┬────────╝
         │                 │
    node_logits(B,N)  veh_logits(B,K)
         │                 │
         └────────┬────────┘
                  │
              Outer Sum
              Reshape
                  │
                  ▼
         ╔─────────────────╗
         │ Joint Action    │
         │ Logits          │
         │ (B, N×K)        │
         │ → Categorical   │
         │ → π(a|s)        │
         ╚────────┬────────╝
                  │
       ┌──────────┴──────────┐
       │                     │
    POLICY                VALUE
   (Actions)             Head
                          │
              ┌───────────┴───────────┐
              │                       │
             MLP (3D → hidden → 1)   │
              │                       │
              ▼                       ▼
         Log Probs               V(s) scalar
         Entropy
         Action Sample


KEY COMPONENTS
──────────────
✓ Node Encoder:        Heterogeneous attention for location embeddings
✓ Vehicle Encoder:     Attention for vehicle embeddings
✓ Graph Encoder:       Multi-relational GNN for routing structure
✓ Node Decoder:        Dot-product attention over nodes
✓ Vehicle Decoder:     Dot-product attention over vehicles
✓ Joint Logits:        Bilevel combination (outer sum) of two independent heads
✓ Value Head:          MLP estimator of state value V(s)

CROSS-ATTENTION PATTERN
───────────────────────
Node Decoder sees:     g_node, mean(Z_veh), g_graph
Vehicle Decoder sees:  g_veh, mean(Z_node), g_graph

This implicit cross-attention allows the decoders to be aware of the other
modality without explicit coupling, enabling independent decision-making.
'''


def main():
    parser = argparse.ArgumentParser(description="Visualize GEMAN architecture")
    parser.add_argument("--format", choices=["png", "svg", "pdf", "ascii"],
                        default="ascii", help="Output format")
    parser.add_argument("--output", "-o", type=str, help="Output file path")

    args = parser.parse_args()

    if args.format == "ascii":
        print(generate_simplified_ascii())
        if args.output:
            Path(args.output).write_text(generate_simplified_ascii())
            print(f"\n✓ Saved to {args.output}")
    else:
        try:
            import graphviz
        except ImportError:
            print("Error: graphviz package required for PNG/SVG/PDF output")
            print("Install with: pip install graphviz")
            sys.exit(1)

        dot_content = generate_graphviz_diagram()

        if args.output:
            output_path = str(Path(args.output).with_suffix(""))
        else:
            output_path = "/tmp/geman_architecture"

        g = graphviz.Source(dot_content, format=args.format)
        g.render(output_path, cleanup=True)
        print(f"✓ Generated {args.format.upper()} diagram: {output_path}.{args.format}")


if __name__ == "__main__":
    main()
