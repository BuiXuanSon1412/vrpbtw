import json
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

# -------------------------------------------------
# Arguments (ONLY filename)
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    type=str,
    help="Relative JSON filename, e.g. S042_N5_C_3G_R50.json",
)
parser.add_argument("--subfolder", type=str, help="Subfolder name, e.g N5/")


args = parser.parse_args()

DATA_ROOT = "../data/generated/data"
RESULT_ROOT = "./result"
IMG_ROOT = "./img"

filename = args.filename
subfolder = args.subfolder

files = []

if filename and subfolder:
    print("Either one: filename OR subfolder")
    sys.exit(1)

if filename:
    files.append(filename)
else:
    subfolder_path = os.path.join(RESULT_ROOT, "f", subfolder)
    files = [
        file
        for file in os.listdir(subfolder_path)
        if os.path.isfile((os.path.join(subfolder_path, file)))
    ]


def run(filename: str):
    parts = filename.split("_")
    subfolder = parts[1]
    data_path = os.path.join(DATA_ROOT, subfolder, filename)
    result_path = os.path.join(RESULT_ROOT, "f", subfolder, filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file not found: {result_path}")

    with open(data_path, "r") as f:
        data = json.load(f)

    with open(result_path, "r") as f:
        result = json.load(f)

    # -------------------------------------------------
    # Extract nodes
    # -------------------------------------------------
    depot = data["Config"]["Depot"]
    customers = data["Nodes"]

    node_coords = {0: depot["coord"]}
    node_info = {0: {"type": "DEPOT", "demand": 0}}

    for c in customers:
        node_coords[c["id"]] = c["coord"]
        node_info[c["id"]] = {
            "type": c["type"],
            "demand": c["demand"],
            "tw_h": c["tw_h"],
        }

    end_depot_idx = len(customers) + 1
    node_coords[end_depot_idx] = depot["coord"]
    node_info[end_depot_idx] = {"type": "DEPOT", "demand": 0}

    # -------------------------------------------------
    # Plot setup
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title("Solution Visualization", fontsize=14, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.3)

    # -------------------------------------------------
    # Truck routes
    # -------------------------------------------------
    start = float("inf")
    end = float("-inf")

    for vehicle in result["routes"]:
        route = vehicle["route"]
        arrival = vehicle["arrival"]
        departure = vehicle["departure"]

        for i in range(len(route) - 1):
            x1, y1 = node_coords[route[i]]
            x2, y2 = node_coords[route[i + 1]]

            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="green", lw=2),
            )

        # arrival / departure under nodes
        for nid, arr, dep in zip(route, arrival, departure):
            if nid == 0:
                start = min(start, dep)
            elif nid == end_depot_idx:
                end = max(end, arr)
            else:
                x, y = node_coords[nid]
                tw = node_info[nid]["tw_h"]
                ax.text(
                    x,
                    y - 1.2,
                    f"[{round(tw[0], 2)}, {arr}, {dep}, {round(tw[1], 2)}]",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="black",
                )

    # -------------------------------------------------
    # Drone trips
    # -------------------------------------------------
    for vehicle in result["routes"]:
        for trip in vehicle.get("trips", []):
            trip_route = trip["route"]
            arrival = trip["arrival"]
            departure = trip["departure"]
            for i in range(len(trip_route) - 1):
                x1, y1 = node_coords[trip_route[i]]
                x2, y2 = node_coords[trip_route[i + 1]]

                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="pink",
                        lw=2,
                        linestyle="--",
                    ),
                )

            for i, (nid, arr, dep) in enumerate(zip(trip_route, arrival, departure)):
                if i == 0:
                    start = min(start, dep)
                elif i == len(trip_route) - 1:
                    end = max(end, arr)
                else:
                    tw = node_info[nid]["tw_h"]
                    x, y = node_coords[nid]
                    ax.text(
                        x,
                        y - 1.2,
                        f"[{round(tw[0], 2)}, {arr}, {dep}, {round(tw[1], 2)}]",
                        ha="center",
                        va="top",
                        fontsize=8,
                        color="black",
                    )

    # -------------------------------------------------
    # Draw nodes
    # -------------------------------------------------
    for nid, (x, y) in node_coords.items():
        info = node_info[nid]

        if nid == 0 or nid == end_depot_idx:
            continue

        color = "blue" if info["type"] == "LINEHAUL" else "red"
        sign = "+" if info["type"] == "LINEHAUL" else "-"

        ax.scatter(x, y, s=160, c=color, edgecolors="black", zorder=10)
        ax.text(
            x,
            y,
            str(nid),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=20,
        )
        ax.text(
            x,
            y + 1.2,
            f"{sign}{abs(info['demand'])}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    x, y = node_coords[0]
    ax.scatter(x, y, s=220, c="black", zorder=10)
    ax.text(x, y, "0", color="white", ha="center", va="center", fontweight="bold")
    ax.text(
        x,
        y - 1.2,
        f"[{start}, {end}]",
        ha="center",
        va="top",
        fontsize=8,
        color="black",
    )

    # -------------------------------------------------
    # Legend (outside plot)
    # -------------------------------------------------
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Depot",
            markerfacecolor="black",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Linehaul (+)",
            markerfacecolor="blue",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Backhaul (-)",
            markerfacecolor="red",
            markersize=10,
        ),
        Line2D([0], [0], color="green", lw=2, label="Truck Route"),
        Line2D([0], [0], color="pink", lw=2, linestyle="--", label="Drone Trip"),
    ]

    ax.legend(
        handles=legend_items,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    # -------------------------------------------------
    # Save image
    # -------------------------------------------------
    # Get the directory part of the output path (e.g., './img/N5')
    name = os.path.splitext(filename)[0]
    out_path = os.path.join(IMG_ROOT, subfolder, f"{name}.png")

    out_dir = os.path.dirname(out_path)

    # Ensure the directory and any necessary subdirectories exist
    os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✔ Saved visualization to {out_path}")


for file in files:
    run(file)
