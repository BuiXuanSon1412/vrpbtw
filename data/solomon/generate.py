import argparse
import json
import math
import os
import random
import numpy as np

# 1.  SINH TỌA ĐỘ  (giữ nguyên logic từ generate.py gốc)

def _generate_coords(n: int, max_coord: float, dist_type: str, seed: int) -> list:
    np.random.seed(seed)
    random.seed(seed)
    
    if dist_type == "R":
        return np.random.uniform(0, max_coord, size=(n, 2)).tolist()

    # Tạo các tâm cụm ngẫu nhiên (đảm bảo khoảng cách tối thiểu 4.sigma giữa các tâm), rồi sinh điểm quanh mỗi tâm bằng Normal(tâm, sigma)
    elif dist_type == "C":
        coords = []
        remaining = n
        centers = []

        # Độ lệch chuẩn của cụm phụ thuộc quy mô bài toán
        if n <= 200:
            std_dev = max_coord / 25
        elif n <= 400:
            std_dev = max_coord / 32
        else:
            std_dev = max_coord / 40

        min_dist = std_dev * 4   # khoảng cách tối thiểu giữa 2 tâm cụm

        while remaining > 0:
            cluster_size = min(remaining, random.randint(10, 15))

            # Tìm tâm cụm mới không chồng lên tâm cũ
            valid = False
            attempts = 0
            proposal = None
            while not valid and attempts < 1000:
                proposal = np.random.uniform(5, max_coord - 5, size=(2,))
                if not centers:
                    valid = True
                else:
                    dists = [np.linalg.norm(proposal - c) for c in centers]
                    if min(dists) >= min_dist:
                        valid = True
                attempts += 1

            centers.append(proposal)
            for _ in range(cluster_size):
                pt = np.random.normal(proposal, std_dev)
                coords.append(np.clip(pt, 0, max_coord).tolist())

            remaining -= cluster_size

        return coords

    # nửa đầu theo C (seed gốc), nửa sau theo R (seed+500)
    elif dist_type == "RC":
        n_c = n // 2
        part_c = _generate_coords(n_c,     max_coord, "C", seed)
        part_r = _generate_coords(n - n_c, max_coord, "R", seed + 500)
        return part_c + part_r

    else:
        raise ValueError(f"dist_type không hợp lệ: {dist_type}")


# 2.  TIỆN ÍCH
def _euclidean(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

# thời gian di chuyển (giờ)
def _travel_time(a, b, speed_kmh: float) -> float:
    return _euclidean(a, b) / speed_kmh


# 3.  SINH CỬA SỔ THỜI GIAN - R/RC

def _tw_random_method(
    coords: list,
    depot_coord: list,
    t_max: float,
    speed_kmh: float,
    service_min: float,
    tw_fraction: float,  # f ∈ {0.25, 0.50, 0.75, 1.00}
    half_width_sigma: float,  # sigma của phân phối chuẩn để sinh nửa độ rộng
    rng: random.Random,
    np_rng: np.random.Generator,
) -> list:
    """
    Sinh TW cho tất cả n khách hàng theo phương pháp R/RC của Solomon.

    Trả về list gồm n phần tử (e_i, l_i).
    Khách hàng không được chọn để có TW → (0, t_max).

    Giải thích:
    Bước 1: Lựa chọn khách hàng có cửa sổ thời gian:
        Xác định tỉ lệ phần trăm khách hàng có cửa sổ thời gian (f)
        Sinh n số theo phân phối đều (0,1) -> tạo hoán vị ngẫu nhiên (i1,...,in).
        Lấy n1 = round(f * n) khách hàng đầu tiên.

    Bước 2: Xác định tâm cửa sổ:
        Tâm m_{i_j} ~ Uniform( e0 + t(0,i_j), l0 - t(i_j,0) - s_{i_j} )
            e0 là thời điểm sớm nhất xe có thể rời kho
            l0 là thời điểm muộn nhất xe phải quay về kho
            t(0, i_j) là thời gian di chuyển từ kho đến khách hàng i_j
            s(i_j) là thời gian phục vụ khách hàng i_j
        => việc chọn tâm trong khoảng này đảm bảo lộ trình khả thi: xe có thể xuất phát từ kho đúng giờ, phục vụ khách hàng, và quay về kho trước l0.

    Bước 3: Xác định độ rộng của cửa sổ thời gian: Để tạo ra độ rộng cho cửa sổ thời gian tại khách hàng i_j -> sinh một nửa độ rộng dưới dạng một số ngẫu nhiên theo chuẩn
        half_w ~ |Normal(0, σ)|   (lấy trị tuyệt đối để luôn ≥ 0)
        e_i = max(e0 + t(0,i), m_i - half_w)
        l_i = min(l0 - t(i,0) - s_i, m_i + half_w)
        Nếu e_i > l_i → dùng cửa sổ đơn điểm [m_i, m_i].
    """
    n = len(coords)
    service_h = service_min / 60.0
    e0 = 0.0
    l0 = t_max

    # hoán vị ngẫu nhiên 
    keys = [rng.random() for _ in range(n)]
    permutation = sorted(range(n), key=lambda k: keys[k])   # i_1, …, i_n
    n1 = round(tw_fraction * n)
    tw_set = set(permutation[:n1])   # tập chỉ số có TW

    # sinh TW 
    tws = []
    for idx in range(n):
        coord = coords[idx]
        t_0_i = _travel_time(depot_coord, coord, speed_kmh)
        t_i_0 = _travel_time(coord, depot_coord, speed_kmh)

        if idx not in tw_set:
            # Không có ràng buộc -> cửa sổ mở hoàn toàn
            tws.append((round(e0, 4), round(l0, 4)))
            continue

        # Khoảng hợp lệ để chọn tâm
        center_lo = e0 + t_0_i      # xe sớm nhất có thể đến
        center_hi = l0 - t_i_0 - service_h      # xe phải rời đi đủ sớm để về kịp

        if center_lo >= center_hi:
            # Khách hàng quá xa
            tws.append((round(center_lo, 4), round(min(center_lo + 0.5, l0), 4)))
            continue

        # tâm
        m = rng.uniform(center_lo, center_hi)

        # nửa độ rộng từ phân phối chuẩn
        half_w = abs(float(np_rng.normal(0, half_width_sigma)))

        e_i = max(center_lo, m - half_w)
        l_i = min(center_hi, m + half_w)

        if e_i > l_i:
            e_i = l_i = m

        tws.append((round(e_i, 4), round(l_i, 4)))

    return tws


# 4.  SINH CỬA SỔ THỜI GIAN - C

def _two_opt_improve(route: list, dist_matrix) -> list:
    """
    2-opt cải thiện đơn giản cho một route (list chỉ số, không gồm depot).
    Trả về route đã được cải thiện.
    """
    n = len(route)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Đổi chiều đoạn [i+1..j]
                old = dist_matrix[route[i]][route[i + 1]] + dist_matrix[route[j]][route[(j + 1) % n]]
                new = dist_matrix[route[i]][route[j]]     + dist_matrix[route[i + 1]][route[(j + 1) % n]]
                if new < old - 1e-9:
                    route[i + 1:j + 1] = route[i + 1:j + 1][::-1]
                    improved = True
    return route


def _build_route_3opt_cluster(
    cluster_indices: list,   # chỉ số node trong cluster (0-based trong coords)
    coords: list,
    depot_coord: list,
) -> list:
    """
    Xây dựng route bằng nearest-neighbor rồi cải thiện bằng 2-opt
    (dùng thay cho 3-opt đầy đủ vì 3-opt phức tạp hơn nhiều; 2-opt
    đủ để tạo ra lịch trình hợp lý theo tinh thần Solomon).

    Trả về thứ tự ghé thăm (list chỉ số trong cluster_indices).

    Giải thích:
        Solomon dùng 3-opt trên từng cụm để xác định lộ trình tự nhiên,
        từ đó làm cơ sở cho tâm cửa sổ thời gian.  Ở đây ta dùng 2-opt
        (đủ mạnh cho cluster nhỏ ≤ 15 node) để giữ code đơn giản mà
        vẫn đúng tinh thần.
    """
    if not cluster_indices:
        return []

    # Ma trận khoảng cách nội bộ (bao gồm depot ở vị trí -1 ngầm định)
    all_pts = [depot_coord] + [coords[i] for i in cluster_indices]
    m = len(all_pts)
    dist_mat = [[_euclidean(all_pts[a], all_pts[b]) for b in range(m)] for a in range(m)]

    # Nearest-neighbor bắt đầu từ depot (index 0 trong dist_mat)
    unvisited = list(range(1, m))   # 1..m-1 tương ứng cluster_indices[0..k-1]
    route = []
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist_mat[cur][x])
        route.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    # 2-opt cải thiện
    route = _two_opt_improve(route, dist_mat)

    # Chuyển lại về chỉ số gốc trong cluster_indices
    # route[k] ∈ {1..m-1} → cluster_indices[route[k]-1]
    return [cluster_indices[r - 1] for r in route]


def _tw_clustered_method(
    coords: list,
    depot_coord: list,
    t_max: float,
    speed_kmh: float,
    service_min: float,
    tw_fraction: float,
    half_width_sigma: float,
    cluster_labels: list,    # list độ dài n, giá trị là nhãn cụm 
    rng: random.Random,
    np_rng: np.random.Generator,
) -> list:
    """
    Giải thích:
    
    Bước 1: Tạo lộ trình cơ sở: Lộ trình 3-opt (thay bằng 2-opt): Tác giả chạy một thuật toán 3-opt trên mỗi cụm khách hàng 
    để xác định các lộ trình di chuyển hợp lý, sau đó chọn một hướng cụ thể cho từng cụm để tạo ra lịch trình
        Với mỗi cụm, chạy nearest-neighbor + 2-opt -> thứ tự ghé thăm.
        Tính thời điểm đến thực tế tại từng node dựa trên lộ trình đó.

    Bước 2: Xác định tâm = thời điểm đến:
        Tâm cửa sổ m_i = arrival_time[i] (dựa trên lộ trình 3-opt), (R là chọn ngẫu nhiên).

    Bước 3: Độ rộng và mật độ (tỷ lệ f):
        Giống phương pháp R: chọn f*n khách hàng (theo hoán vị ngẫu nhiên), sinh nửa độ rộng từ |N(0,σ)|.
    """
    n = len(coords)
    service_h = service_min / 60.0
    e0 = 0.0
    l0 = t_max

    # Tính arrival_time cho từng node qua lộ trình per-cluster 
    arrival = [None] * n

    unique_labels = sorted(set(cluster_labels))
    for label in unique_labels:
        members = [i for i in range(n) if cluster_labels[i] == label]
        if not members:
            continue

        ordered = _build_route_3opt_cluster(members, coords, depot_coord)

        cur_pos = depot_coord
        cur_time = e0
        for idx in ordered:
            t_travel = _travel_time(cur_pos, coords[idx], speed_kmh)
            arrive = cur_time + t_travel
            # Chờ nếu đến sớm (không áp dụng TW lúc này, chỉ tính thời điểm)
            arrival[idx] = arrive
            cur_time = arrive + service_h
            cur_pos = coords[idx]

    # Node nào không có arrival (lỗi label) → fallback
    for i in range(n):
        if arrival[i] is None:
            arrival[i] = _travel_time(depot_coord, coords[i], speed_kmh)

    #Hoán vị -> chọn tw_set (giống R) 
    keys = [rng.random() for _ in range(n)]
    permutation = sorted(range(n), key=lambda k: keys[k])
    n1 = round(tw_fraction * n)
    tw_set = set(permutation[:n1])

    # Sinh TW
    tws = []
    for idx in range(n):
        coord = coords[idx]
        t_0_i = _travel_time(depot_coord, coord, speed_kmh)
        t_i_0 = _travel_time(coord, depot_coord, speed_kmh)

        center_lo = e0 + t_0_i
        center_hi = l0 - t_i_0 - service_h

        if idx not in tw_set:
            tws.append((round(e0, 4), round(l0, 4)))
            continue

        if center_lo >= center_hi:
            tws.append((round(center_lo, 4), round(min(center_lo + 0.5, l0), 4)))
            continue

        # Tâm = thời điểm đến từ lộ trình 
        m = max(center_lo, min(center_hi, arrival[idx]))

        # Nửa độ rộng từ phân phối chuẩn
        half_w = abs(float(np_rng.normal(0, half_width_sigma)))

        e_i = max(center_lo, m - half_w)
        l_i = min(center_hi, m + half_w)

        if e_i > l_i:
            e_i = l_i = m

        tws.append((round(e_i, 4), round(l_i, 4)))

    return tws


# 5.  XÁC ĐỊNH NHÃN CỤM (dùng cho C)

def _assign_cluster_labels(
    coords: list,
    dist_type: str,
    seed: int,
    max_coord: float,
    n: int,
) -> list:
    """
    Gán nhãn cụm cho mỗi node.
    - R / RC : mỗi node là một cụm riêng (không dùng phương pháp C)
    - C      : nhóm các node liền kề (nearest-neighbor clustering đơn giản, phù hợp với cách sinh coords theo cụm).

    Với dist_type == "C", các node được sinh theo từng cụm tuần tự (10-15
    node mỗi cụm), nên ta chỉ cần gán nhãn theo block.
    """
    if dist_type != "C":
        # Phương pháp R/RC không cần nhãn cụm thật sự
        return list(range(n))

    # Với C: block-assign theo kích thước cụm 10-15 (khớp với _generate_coords)
    rng = random.Random(seed)
    labels = []
    label = 0
    remaining = n
    while remaining > 0:
        size = min(remaining, rng.randint(10, 15))
        labels.extend([label] * size)
        label += 1
        remaining -= size
    return labels[:n]


# 6.  Main

def create_instance(
    config: dict,
    n: int,
    dist_type: str,
    ratio: float,           # tỷ lệ linehaul (0.5 → 50% linehaul)
    seed: int,
    tw_fraction: float = 1.0,            # f ∈ {0.25, 0.50, 0.75, 1.00}
    half_width_sigma: float | None = None,  # sigma nửa độ rộng TW (giờ)
) -> dict:
    """
    Tạo một instance VRP-BLTW theo chuẩn Solomon.

    Tham số
    -------
    config          : dict từ config.json
    n               : số khách hàng
    dist_type       : "R" | "C" | "RC"
    ratio           : tỷ lệ khách hàng linehaul
    seed            : seed ngẫu nhiên
    tw_fraction     : f — tỷ lệ khách hàng có ràng buộc TW thật sự
    half_width_sigma: σ của phân phối chuẩn sinh nửa độ rộng TW (giờ).
                      Nếu None → tự tính = t_max * 0.08 (khoảng 1h cho 12h horizon)
    """
    # Khởi tạo RNG 
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    max_coord    = config["MAX_COORD"]
    speed_truck  = config["V_TRUCK_KM_H"]
    service_min  = config["SERVICE_TIME_MIN"]
    t_max        = float(config.get("T_MAX_SYSTEM_H", 12.0))
    num_trucks   = config["FLEET_SIZES"].get(str(n), max(1, n // 8))
    depot_coord  = [max_coord / 2, max_coord / 2]

    if half_width_sigma is None:
        # Mặc định: sigma ≈ 8% horizon (~1 giờ với 12h horizon)
        half_width_sigma = t_max * 0.08

    # Sinh tọa độ
    coords = _generate_coords(n, max_coord, dist_type, seed)

    # Gán demand & type
    n_linehaul = int(n * ratio)
    types = ["LINEHAUL"] * n_linehaul + ["BACKHAUL"] * (n - n_linehaul)
    rng.shuffle(types)

    demands = []
    for t in types:
        if t == "LINEHAUL":
            d = rng.randint(config["DEMAND_RANGE_LINEHAUL"][0],
                            config["DEMAND_RANGE_LINEHAUL"][1])
        else:
            d = rng.randint(config["DEMAND_RANGE_BACKHAUL"][0],
                            config["DEMAND_RANGE_BACKHAUL"][1])
        demands.append(d)

    # Xác định nhãn cụm (chỉ cần cho C) 
    cluster_labels = _assign_cluster_labels(coords, dist_type, seed, max_coord, n)

    # Sinh cửa sổ thời gian 
    if dist_type == "C":
        # Phương pháp C: tâm TW = thời điểm đến theo lộ trình 3-opt
        tws = _tw_clustered_method(
            coords, depot_coord, t_max, speed_truck, service_min,
            tw_fraction, half_width_sigma, cluster_labels, rng, np_rng,
        )
    else:
        # Phương pháp R / RC: tâm TW ngẫu nhiên trong khoảng khả thi
        tws = _tw_random_method(
            coords, depot_coord, t_max, speed_truck, service_min,
            tw_fraction, half_width_sigma, rng, np_rng,
        )

    # Đóng gói nodes
    nodes = []
    for i in range(n):
        e_i, l_i = tws[i]
        nodes.append({
            "id"    : i + 1,
            "coord" : coords[i],
            "demand": demands[i],
            "type"  : types[i],
            "tw_h"  : [e_i, l_i],
        })

    num_clusters = max(1, n // 15) if dist_type != "R" else 0

    return {
        "Config": {
            "General": {
                "NUM_CUSTOMERS"            : n,
                "NUM_NODES"                : n + 1,
                "MAX_COORD_KM"             : max_coord,
                "T_MAX_SYSTEM_H"           : t_max,
                "TIME_WINDOW_SCALING_FACTOR": config.get("TIME_WINDOW_SCALING_FACTOR", 1.5),
                "COORD_DISTRIBUTION"       : dist_type,
                "NUM_CLUSTERS"             : num_clusters,
                "TW_FRACTION"              : tw_fraction,
                "HALF_WIDTH_SIGMA_H"       : round(half_width_sigma, 4),
            },
            "Vehicles": {
                "NUM_TRUCKS"      : num_trucks,
                "NUM_DRONES"      : num_trucks,
                "V_TRUCK_KM_H"    : speed_truck,
                "V_DRONE_KM_H"    : config["V_DRONE_KM_H"],
                "CAPACITY_TRUCK"  : config["CAPACITY_TRUCK"],
                "CAPACITY_DRONE"  : config["CAPACITY_DRONE"],
                "DRONE_TAKEOFF_MIN": config["DRONE_TAKEOFF_MIN"],
                "DRONE_LANDING_MIN": config["DRONE_LANDING_MIN"],
                "SERVICE_TIME_MIN": service_min,
                "DRONE_DURATION_H": config["DRONE_DURATION_H"],
            },
            "Depot": {
                "id"          : 0,
                "coord"       : depot_coord,
                "time_window_h": [0.0, t_max],
            },
        },
        "Nodes": nodes,
    }

if __name__ == "__main__":
    """
    # R, 10 node, 5 instance, f=100%
    python generate.py --n 10 --dist R --count 5

    # C, 50 node, f=75%
    python generate.py --n 50 --dist C --tw_fraction 0.75

    # RC, 100 node, f=50%, σ=0.5h
    python generate.py --n 100 --dist RC --tw_fraction 0.50 --sigma 0.5
    """
    parser = argparse.ArgumentParser(
        description="Sinh dữ liệu VRP-BLTW theo chuẩn Solomon"
    )
    parser.add_argument(
        "--n", type=int,
        choices=[10, 20, 50, 100, 200, 400, 1000], default=10,
        help="Số khách hàng",
    )
    parser.add_argument(
        "--dist", choices=["R", "C", "RC"], default="RC",
        help="Kiểu phân phối tọa độ",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.5,
        help="Tỷ lệ khách hàng linehaul (0..1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed ngẫu nhiên (instance đầu tiên)",
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Số instance cần sinh",
    )
    parser.add_argument(
        "--tw_fraction", type=float, default=1.0,
        choices=[0.25, 0.50, 0.75, 1.00],
        help="Tỷ lệ khách hàng có ràng buộc TW thật sự (f)",
    )
    parser.add_argument(
        "--sigma", type=float, default=None,
        help="Nửa độ rộng TW σ (giờ). Mặc định: 8%% của T_MAX",
    )
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="Đường dẫn tới file config.json",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Thư mục đầu ra. Mặc định: data/N{n}/",
    )
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(
        os.path.join(BASE_DIR, "..", "generated", "config.json"),
        "r"
    ) as f:
        cfg = json.load(f)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    out_dir = args.outdir or os.path.join(BASE_DIR, 'data', f'N{args.n}')

    os.makedirs(out_dir, exist_ok=True)

    for i in range(args.count):
        s = args.seed + i
        data = create_instance(
            cfg, args.n, args.dist, args.ratio, s,
            tw_fraction=args.tw_fraction,
            half_width_sigma=args.sigma,
        )
        fname = (
            f"S{s:03d}_N{args.n}_{args.dist}"
            f"_R{int(args.ratio * 100)}"
            f"_TW{int(args.tw_fraction * 100)}.json"
        )
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Generated: {fname}")