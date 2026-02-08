import env_factory
import time
import math
import random
import heapq
import sys
import select
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from matplotlib.patches import Polygon as MplPolygon

SHOW_PLOTS = False

class GridMapper:
    def __init__(self, setup, resolution=0.1):
        self.res = resolution
        self.bounds = setup['map_bounds'] # [-10, 10, -10, 10]
        self.width = int((self.bounds[1] - self.bounds[0]) / self.res)
        self.height = int((self.bounds[3] - self.bounds[2]) / self.res)
        self.grid = np.zeros((self.width, self.height))

    def world_to_grid(self, x, y):
        """Converts continuous world coordinates to discrete grid indices."""
        gx = int(round((x - self.bounds[0]) / self.res))
        gy = int(round((y - self.bounds[2]) / self.res))
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Converts discrete grid indices to continuous world coordinates."""
        x = gx * self.res + self.bounds[0] + self.res/2
        y = gy * self.res + self.bounds[2] + self.res/2
        return x, y

    def fill_obstacles(self, obstacles):
        """Fills the grid with static obstacles from the setup."""
        for obs in obstacles:
            # Get corners in world coordinates
            x_min, x_max = obs['pos'][0] - obs['extents'][0], obs['pos'][0] + obs['extents'][0]
            y_min, y_max = obs['pos'][1] - obs['extents'][1], obs['pos'][1] + obs['extents'][1]
            
            # Convert to grid indices
            gx1, gy1 = self.world_to_grid(x_min, y_min)
            gx2, gy2 = self.world_to_grid(x_max, y_max)
            
            # Fill the rectangle in the grid
            self.grid[max(0, gx1):min(self.width, gx2), max(0, gy1):min(self.height, gy2)] = 1

    def compute_cspace(self, obstacles, robot_type, robot_geometry):
        """Task 1.1: Compute C-space obstacles via Minkowski sum.
        
        For each obstacle, expand it by the robot's footprint (reflected about the origin)
        to get the C-space obstacle. The result is stored in self.cspace_grid.
        """
        self.cspace_grid = np.zeros((self.width, self.height))

        # Compute the robot's "radius" for Minkowski sum
        if robot_type == "RECT":
            # Robot is a rectangle with dimensions [width, length]
            robot_half_w = robot_geometry[0] / 2.0
            robot_half_l = robot_geometry[1] / 2.0
            # For a rectangle centered at origin, -R = R (symmetric)
            # Minkowski sum of two axis-aligned rectangles is another rectangle
            for obs in obstacles:
                ox, oy = obs['pos'][0], obs['pos'][1]
                ex, ey = obs['extents'][0], obs['extents'][1]

                # Expanded obstacle in C-space
                cspace_x_min = ox - ex - robot_half_w
                cspace_x_max = ox + ex + robot_half_w
                cspace_y_min = oy - ey - robot_half_l
                cspace_y_max = oy + ey + robot_half_l

                gx1, gy1 = self.world_to_grid(cspace_x_min, cspace_y_min)
                gx2, gy2 = self.world_to_grid(cspace_x_max, cspace_y_max)

                self.cspace_grid[
                    max(0, gx1):min(self.width, gx2),
                    max(0, gy1):min(self.height, gy2)
                ] = 1
        else:
            # robot_type == "POLY"
            # Reflect the robot polygon about the origin: -R
            neg_robot = [[-pt[0], -pt[1]] for pt in robot_geometry]

            for obs in obstacles:
                ox, oy = obs['pos'][0], obs['pos'][1]
                ex, ey = obs['extents'][0], obs['extents'][1]

                # Obstacle rectangle corners (CCW)
                obs_corners = [
                    [ox - ex, oy - ey],
                    [ox + ex, oy - ey],
                    [ox + ex, oy + ey],
                    [ox - ex, oy + ey],
                ]

                # Minkowski sum of convex obstacle rect and reflected robot polygon
                cspace_polygon = self._minkowski_sum_convex(obs_corners, neg_robot)

                # Rasterise the C-space polygon onto the grid
                self._fill_polygon(self.cspace_grid, cspace_polygon)

        # Also mark the map boundary as C-space obstacle
        # The robot center cannot be within robot_radius of the boundary
        if robot_type == "RECT":
            margin_x = robot_half_w
            margin_y = robot_half_l
        else:
            # Use the max extent of the polygon as margin
            pts = np.array(robot_geometry)
            margin_x = np.max(np.abs(pts[:, 0]))
            margin_y = np.max(np.abs(pts[:, 1]))

        # Mark boundary margins
        bx_min = self.bounds[0] + margin_x
        bx_max = self.bounds[1] - margin_x
        by_min = self.bounds[2] + margin_y
        by_max = self.bounds[3] - margin_y

        gx_lo, gy_lo = self.world_to_grid(bx_min, by_min)
        gx_hi, gy_hi = self.world_to_grid(bx_max, by_max)

        # Left boundary
        self.cspace_grid[0:max(0, gx_lo), :] = 1
        # Right boundary
        self.cspace_grid[min(self.width, gx_hi):self.width, :] = 1
        # Bottom boundary
        self.cspace_grid[:, 0:max(0, gy_lo)] = 1
        # Top boundary
        self.cspace_grid[:, min(self.height, gy_hi):self.height] = 1

        return self.cspace_grid

    def _minkowski_sum_convex(self, P, Q):
        """Compute the Minkowski sum of two convex polygons P and Q.
        
        Uses the rotating calipers / sorted-edge-merge algorithm.
        Both polygons must be in CCW order.
        """
        def ensure_ccw(poly):
            """Ensure polygon vertices are in counter-clockwise order."""
            area = 0
            n = len(poly)
            for i in range(n):
                j = (i + 1) % n
                area += poly[i][0] * poly[j][1]
                area -= poly[j][0] * poly[i][1]
            if area < 0:
                poly.reverse()
            return poly

        def edge_angles(poly):
            """Compute edge vectors and their angles for a CCW polygon."""
            n = len(poly)
            edges = []
            angles = []
            for i in range(n):
                j = (i + 1) % n
                dx = poly[j][0] - poly[i][0]
                dy = poly[j][1] - poly[i][1]
                edges.append((dx, dy))
                angles.append(np.arctan2(dy, dx))
            return edges, angles

        P = ensure_ccw([list(p) for p in P])
        Q = ensure_ccw([list(q) for q in Q])

        # Start from the bottom-most (then left-most) vertex
        def bottom_left_idx(poly):
            idx = 0
            for i in range(1, len(poly)):
                if poly[i][1] < poly[idx][1] or (poly[i][1] == poly[idx][1] and poly[i][0] < poly[idx][0]):
                    idx = i
            return idx

        pi = bottom_left_idx(P)
        qi = bottom_left_idx(Q)

        # Reorder so that bottom-left vertex is first
        P = P[pi:] + P[:pi]
        Q = Q[qi:] + Q[:qi]

        p_edges, p_angles = edge_angles(P)
        q_edges, q_angles = edge_angles(Q)

        result = []
        i, j = 0, 0
        n_p, n_q = len(P), len(Q)

        # Starting vertex of the Minkowski sum
        result.append([P[0][0] + Q[0][0], P[0][1] + Q[0][1]])

        while i < n_p or j < n_q:
            if i >= n_p:
                edge = q_edges[j]
                j += 1
            elif j >= n_q:
                edge = p_edges[i]
                i += 1
            else:
                # Compare angles; advance the polygon with the smaller edge angle
                pa = p_angles[i] % (2 * np.pi)
                qa = q_angles[j] % (2 * np.pi)
                if pa < qa - 1e-10:
                    edge = p_edges[i]
                    i += 1
                elif qa < pa - 1e-10:
                    edge = q_edges[j]
                    j += 1
                else:
                    # Same angle: merge both edges
                    edge = (p_edges[i][0] + q_edges[j][0], p_edges[i][1] + q_edges[j][1])
                    i += 1
                    j += 1

            new_pt = [result[-1][0] + edge[0], result[-1][1] + edge[1]]
            result.append(new_pt)

        # Remove the last point (duplicate of first)
        if len(result) > 1:
            result.pop()

        return result

    def _fill_polygon(self, grid, polygon_pts):
        """Rasterise a polygon onto the grid, setting cells to 1."""
        grid_pts = []
        for pt in polygon_pts:
            gx, gy = self.world_to_grid(pt[0], pt[1])
            grid_pts.append((gx, gy))

        nodes = np.array(grid_pts)
        rr, cc = polygon(nodes[:, 0], nodes[:, 1], shape=(self.width, self.height))
        grid[rr, cc] = 1

    def is_collision_free(self, x, y):
        """Task 1.2: Check if a configuration (x, y) is collision-free in C-space."""
        gx, gy = self.world_to_grid(x, y)
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False  # Out of bounds
        return self.cspace_grid[gx, gy] == 0

    def overlay_robot(self, x, y, robot_type, robot_geo):
        temp_grid = self.grid.copy()
        if robot_type == "RECT":
            # robot_geo is [width, length]
            hw, hl = robot_geo[0]/2, robot_geo[1]/2
            gx1, gy1 = self.world_to_grid(x - hw, y - hl)
            gx2, gy2 = self.world_to_grid(x + hw, y + hl)
            temp_grid[max(0, gx1):min(self.width, gx2), max(0, gy1):min(self.height, gy2)] = 2
        else:
            # robot_type == "POLY"
            poly_grid_pts = []
            for pt in robot_geo:
                gx, gy = self.world_to_grid(x + pt[0], y + pt[1])
                poly_grid_pts.append((gx, gy))

            nodes = np.array(poly_grid_pts)
            min_x, min_y = nodes.min(axis=0)
            max_x, max_y = nodes.max(axis=0)

            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(self.width - 1, max_x)
            max_y = min(self.height - 1, max_y)

            from matplotlib.path import Path
            path = Path(poly_grid_pts)

            for ix in range(int(min_x), int(max_x) + 1):
                for iy in range(int(min_y), int(max_y) + 1):
                    if path.contains_point((ix, iy)):
                        temp_grid[ix, iy] = 2
        return temp_grid


def octile_distance(a, b):
    """Admissible heuristic for 8-connected grids."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1.0) * min(dx, dy)


def reconstruct_grid_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def compute_path_length(path):
    if not path or len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        length += math.hypot(dx, dy)
    return length


def astar_search(mapper, start_xy, goal_xy, weight=1.0):
    """Task 2.1: A* / Weighted A* on 8-connected occupancy grid."""
    t0 = time.perf_counter()
    grid = mapper.cspace_grid
    width, height = grid.shape

    sx, sy = mapper.world_to_grid(start_xy[0], start_xy[1])
    gx, gy = mapper.world_to_grid(goal_xy[0], goal_xy[1])
    start = (sx, sy)
    goal = (gx, gy)

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    if not in_bounds(sx, sy) or not in_bounds(gx, gy):
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "path": None,
            "time_ms": elapsed,
            "path_length": math.inf,
            "nodes_expanded": 0,
            "success": False,
            "reason": "start_or_goal_out_of_bounds"
        }
    if grid[sx, sy] == 1 or grid[gx, gy] == 1:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "path": None,
            "time_ms": elapsed,
            "path_length": math.inf,
            "nodes_expanded": 0,
            "success": False,
            "reason": "start_or_goal_in_collision"
        }

    g_score = np.full((width, height), np.inf, dtype=float)
    closed = np.zeros((width, height), dtype=bool)
    came_from = {}
    g_score[sx, sy] = 0.0

    # (dx, dy, movement_cost)
    neighbors = [
        (1, 0, mapper.res), (-1, 0, mapper.res),
        (0, 1, mapper.res), (0, -1, mapper.res),
        (1, 1, mapper.res * math.sqrt(2)), (1, -1, mapper.res * math.sqrt(2)),
        (-1, 1, mapper.res * math.sqrt(2)), (-1, -1, mapper.res * math.sqrt(2))
    ]

    open_heap = []
    counter = 0
    h0 = octile_distance(start, goal) * mapper.res
    heapq.heappush(open_heap, (g_score[sx, sy] + weight * h0, counter, start))

    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        cx, cy = current

        if closed[cx, cy]:
            continue

        closed[cx, cy] = True
        nodes_expanded += 1

        if current == goal:
            grid_path = reconstruct_grid_path(came_from, current)
            world_path = [mapper.grid_to_world(px, py) for px, py in grid_path]
            elapsed = (time.perf_counter() - t0) * 1000.0
            return {
                "path": world_path,
                "time_ms": elapsed,
                "path_length": compute_path_length(world_path),
                "nodes_expanded": nodes_expanded,
                "success": True
            }

        for dx, dy, move_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny):
                continue
            if closed[nx, ny]:
                continue
            if grid[nx, ny] == 1:
                continue

            tentative_g = g_score[cx, cy] + move_cost
            if tentative_g < g_score[nx, ny]:
                g_score[nx, ny] = tentative_g
                came_from[(nx, ny)] = (cx, cy)
                h = octile_distance((nx, ny), goal) * mapper.res
                counter += 1
                heapq.heappush(open_heap, (tentative_g + weight * h, counter, (nx, ny)))

    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "path": None,
        "time_ms": elapsed,
        "path_length": math.inf,
        "nodes_expanded": nodes_expanded,
        "success": False,
        "reason": "no_path_found"
    }


def feasibility_check(mapper, start_xy, goal_xy):
    """
    Edge-case handling: verify if the random layout is solvable.
    Uses A* as a deterministic reachability check.
    """
    res = astar_search(mapper, start_xy, goal_xy, weight=1.0)
    return {
        "feasible": bool(res["success"]),
        "time_ms": float(res["time_ms"]),
        "nodes_expanded": int(res.get("nodes_expanded", 0)),
        "reason": res.get("reason", "" if res["success"] else "unknown"),
        "path_length": float(res["path_length"]) if res["success"] else math.inf,
    }


class RRTTree:
    def __init__(self, root):
        self.points = [np.array(root, dtype=float)]
        self.parents = [-1]

    def add(self, point, parent_idx):
        self.points.append(np.array(point, dtype=float))
        self.parents.append(parent_idx)
        return len(self.points) - 1

    def nearest_index(self, query_point):
        query_point = np.array(query_point, dtype=float)
        best_idx = 0
        best_dist = float("inf")
        for idx, pt in enumerate(self.points):
            d2 = np.sum((pt - query_point) ** 2)
            if d2 < best_dist:
                best_dist = d2
                best_idx = idx
        return best_idx


def line_of_sight_collision_free(p0, p1, mapper, sample_step=None):
    """Task 2.2: Simple segment collision check by interpolation sampling."""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    if sample_step is None:
        sample_step = mapper.res * 0.5

    segment = p1 - p0
    dist = np.linalg.norm(segment)
    if dist < 1e-12:
        return mapper.is_collision_free(p0[0], p0[1])

    num_steps = max(1, int(math.ceil(dist / sample_step)))
    for i in range(num_steps + 1):
        t = i / num_steps
        x = p0[0] + t * segment[0]
        y = p0[1] + t * segment[1]
        if not mapper.is_collision_free(x, y):
            return False
    return True


def steer(from_pt, to_pt, step_size):
    from_pt = np.array(from_pt, dtype=float)
    to_pt = np.array(to_pt, dtype=float)
    delta = to_pt - from_pt
    dist = np.linalg.norm(delta)
    if dist <= step_size:
        return to_pt.copy()
    return from_pt + (delta / dist) * step_size


def rrt_extend(tree, target, step_size, mapper):
    nearest_idx = tree.nearest_index(target)
    q_near = tree.points[nearest_idx]
    q_new = steer(q_near, target, step_size)

    if not line_of_sight_collision_free(q_near, q_new, mapper):
        return "trapped", None

    new_idx = tree.add(q_new, nearest_idx)

    if np.linalg.norm(tree.points[new_idx] - np.array(target)) < 1e-9:
        return "reached", new_idx
    return "advanced", new_idx


def rrt_connect(tree, target, step_size, mapper):
    status = "advanced"
    new_idx = None
    while status == "advanced":
        status, new_idx = rrt_extend(tree, target, step_size, mapper)
    return status, new_idx


def trace_to_root(tree, node_idx):
    chain = []
    idx = node_idx
    while idx != -1:
        chain.append(tree.points[idx])
        idx = tree.parents[idx]
    return chain


def merge_connected_paths(start_tree, start_idx, goal_tree, goal_idx):
    # Each trace is [connection, ..., root]. Convert to [root, ..., connection].
    from_start = list(reversed(trace_to_root(start_tree, start_idx)))
    from_goal = trace_to_root(goal_tree, goal_idx)

    # Avoid duplicating the same connection point.
    merged = from_start + from_goal[1:]
    return [(float(p[0]), float(p[1])) for p in merged]


def rrt_connect_planner(
    mapper,
    start_xy,
    goal_xy,
    bounds,
    step_size=0.5,
    max_iterations=10000,
    goal_bias=0.1,
    rng_seed=0,
):
    """Task 2.2: RRT-Connect in continuous C-space."""
    t0 = time.perf_counter()
    rng = random.Random(rng_seed)

    start = np.array(start_xy, dtype=float)
    goal = np.array(goal_xy, dtype=float)

    if not mapper.is_collision_free(start[0], start[1]) or not mapper.is_collision_free(goal[0], goal[1]):
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "path": None,
            "time_ms": elapsed,
            "path_length": math.inf,
            "vertices_sampled": 0,
            "success": False,
            "reason": "start_or_goal_in_collision"
        }

    start_tree = RRTTree(start)
    goal_tree = RRTTree(goal)

    tree_a = start_tree
    tree_b = goal_tree
    swapped = False

    for _ in range(max_iterations):
        if rng.random() < goal_bias:
            q_rand = goal if not swapped else start
        else:
            q_rand = np.array([
                rng.uniform(bounds[0], bounds[1]),
                rng.uniform(bounds[2], bounds[3])
            ], dtype=float)
            if not mapper.is_collision_free(q_rand[0], q_rand[1]):
                continue

        status_a, idx_a = rrt_extend(tree_a, q_rand, step_size, mapper)
        if status_a != "trapped":
            q_new = tree_a.points[idx_a]
            status_b, idx_b = rrt_connect(tree_b, q_new, step_size, mapper)

            if status_b == "reached":
                if not swapped:
                    path = merge_connected_paths(start_tree, idx_a, goal_tree, idx_b)
                else:
                    path = merge_connected_paths(start_tree, idx_b, goal_tree, idx_a)

                elapsed = (time.perf_counter() - t0) * 1000.0
                return {
                    "path": path,
                    "time_ms": elapsed,
                    "path_length": compute_path_length(path),
                    "vertices_sampled": len(start_tree.points) + len(goal_tree.points),
                    "success": True
                }

        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped

    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "path": None,
        "time_ms": elapsed,
        "path_length": math.inf,
        "vertices_sampled": len(start_tree.points) + len(goal_tree.points),
        "success": False,
        "reason": "no_path_found"
    }


def rrt_connect_with_narrow_passage_fallback(
    mapper,
    start_xy,
    goal_xy,
    bounds,
    step_size=0.5,
    max_iterations=10000,
    goal_bias=0.1,
    rng_seed=0,
):
    """
    Edge-case handling: for narrow passages, retry RRT-Connect with tighter step-size
    and more iterations when the default run fails.
    """
    primary = rrt_connect_planner(
        mapper=mapper,
        start_xy=start_xy,
        goal_xy=goal_xy,
        bounds=bounds,
        step_size=step_size,
        max_iterations=max_iterations,
        goal_bias=goal_bias,
        rng_seed=rng_seed,
    )
    if primary["success"]:
        primary["strategy"] = "default"
        return primary

    fallback = rrt_connect_planner(
        mapper=mapper,
        start_xy=start_xy,
        goal_xy=goal_xy,
        bounds=bounds,
        step_size=max(0.2, step_size * 0.6),
        max_iterations=int(max_iterations * 2.2),
        goal_bias=min(0.25, goal_bias * 1.5),
        rng_seed=rng_seed + 7919,
    )
    if fallback["success"]:
        fallback["strategy"] = "narrow_passage_fallback"
        fallback["fallback_trigger"] = primary.get("reason", "primary_failed")
        return fallback

    primary["strategy"] = "default_and_fallback_failed"
    return primary


def print_phase2_comparison_table(results):
    headers = [
        "Algorithm",
        "Time to 1st Solution (ms)",
        "Path Length (m)",
        "Memory Usage"
    ]
    rows = []
    for name, res in results.items():
        if res["success"]:
            time_str = f"{res['time_ms']:.2f}"
            length_str = f"{res['path_length']:.2f}"
            if "nodes_expanded" in res:
                mem_str = f"{res['nodes_expanded']} expanded"
            else:
                mem_str = f"{res['vertices_sampled']} vertices"
        else:
            time_str = f"{res['time_ms']:.2f}"
            length_str = "N/A"
            mem_str = (
                f"{res.get('nodes_expanded', res.get('vertices_sampled', 0))} visited"
            )
        rows.append([name, time_str, length_str, mem_str])

    col_widths = []
    for col_idx in range(len(headers)):
        values = [headers[col_idx]] + [r[col_idx] for r in rows]
        col_widths.append(max(len(v) for v in values) + 2)

    def fmt_row(vals):
        return "".join(v.ljust(col_widths[i]) for i, v in enumerate(vals))

    print("\n=== Phase 2 Comparison (Task 2.3) ===")
    print(fmt_row(headers))
    print(fmt_row(["-" * (w - 2) for w in col_widths]))
    for r in rows:
        print(fmt_row(r))


def run_phase2_algorithms_for_seed(
    seed,
    resolution=0.1,
    rrt_step_size=0.5,
    rrt_max_iterations=15000,
    rrt_goal_bias=0.1,
):
    """Run A* and RRT-Connect once for one random seed in DIRECT mode."""
    env = env_factory.RandomizedWarehouse(seed=seed, mode=env_factory.p.DIRECT)
    setup = env.get_problem_setup()

    mapper = GridMapper(setup, resolution=resolution)
    mapper.fill_obstacles(setup["static_obstacles"])
    mapper.compute_cspace(setup["static_obstacles"], setup["robot_type"], setup["robot_geometry"])

    a_star_result = astar_search(mapper, setup["start"], setup["goal"], weight=1.0)
    feasible = bool(a_star_result["success"])
    if feasible:
        rrt_result = rrt_connect_with_narrow_passage_fallback(
            mapper=mapper,
            start_xy=setup["start"],
            goal_xy=setup["goal"],
            bounds=setup["map_bounds"],
            step_size=rrt_step_size,
            max_iterations=rrt_max_iterations,
            goal_bias=rrt_goal_bias,
            rng_seed=seed,
        )
    else:
        rrt_result = {
            "path": None,
            "time_ms": 0.0,
            "path_length": math.inf,
            "vertices_sampled": 0,
            "success": False,
            "reason": "map_infeasible_by_astar_check",
            "strategy": "skipped_due_to_infeasible_map",
        }

    return {
        "seed": int(seed),
        "feasible": feasible,
        "A*": a_star_result,
        "RRT-Connect": rrt_result,
    }


def print_task23_multiseed_table(multiseed_results):
    """Print Task 2.3 comparison table across multiple seeds."""
    headers = [
        "Seed",
        "Feasible",
        "A* Time (ms)",
        "A* Length (m)",
        "A* Memory",
        "RRT Time (ms)",
        "RRT Length (m)",
        "RRT Memory",
        "RRT Strategy",
    ]
    rows = []
    for item in multiseed_results:
        a_res = item["A*"]
        r_res = item["RRT-Connect"]
        rows.append(
            [
                str(item["seed"]),
                ("Y" if item.get("feasible", False) else "N"),
                f"{a_res['time_ms']:.2f}",
                f"{a_res['path_length']:.2f}" if a_res["success"] else "N/A",
                str(a_res.get("nodes_expanded", 0)),
                f"{r_res['time_ms']:.2f}",
                f"{r_res['path_length']:.2f}" if r_res["success"] else "N/A",
                str(r_res.get("vertices_sampled", 0)),
                str(r_res.get("strategy", "default")),
            ]
        )

    # Mean row (success-aware for path length).
    if rows:
        a_times = [item["A*"]["time_ms"] for item in multiseed_results]
        a_lens = [item["A*"]["path_length"] for item in multiseed_results if item["A*"]["success"]]
        a_mems = [item["A*"].get("nodes_expanded", 0) for item in multiseed_results]
        r_times = [item["RRT-Connect"]["time_ms"] for item in multiseed_results]
        r_lens = [item["RRT-Connect"]["path_length"] for item in multiseed_results if item["RRT-Connect"]["success"]]
        r_mems = [item["RRT-Connect"].get("vertices_sampled", 0) for item in multiseed_results]
        rows.append(
            [
                "MEAN",
                "-",
                f"{float(np.mean(a_times)):.2f}",
                f"{float(np.mean(a_lens)):.2f}" if a_lens else "N/A",
                f"{float(np.mean(a_mems)):.1f}",
                f"{float(np.mean(r_times)):.2f}",
                f"{float(np.mean(r_lens)):.2f}" if r_lens else "N/A",
                f"{float(np.mean(r_mems)):.1f}",
                "-",
            ]
        )

    col_widths = []
    for col_idx in range(len(headers)):
        values = [headers[col_idx]] + [r[col_idx] for r in rows]
        col_widths.append(max(len(v) for v in values) + 2)

    def fmt_row(vals):
        return "".join(v.ljust(col_widths[i]) for i, v in enumerate(vals))

    print("\n=== Task 2.3 Multi-Seed Comparison (A* vs RRT-Connect) ===")
    print(fmt_row(headers))
    print(fmt_row(["-" * (w - 2) for w in col_widths]))
    for r in rows:
        print(fmt_row(r))


def save_task23_multiseed_csv(multiseed_results, csv_path="phase2_task23_multiseed.csv"):
    """Save Task 2.3 multi-seed comparison to CSV for report usage."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "feasible_by_astar_check",
                "astar_success",
                "astar_time_ms",
                "astar_path_length_m",
                "astar_nodes_expanded",
                "rrt_success",
                "rrt_time_ms",
                "rrt_path_length_m",
                "rrt_vertices_sampled",
                "rrt_strategy",
            ]
        )
        for item in multiseed_results:
            a_res = item["A*"]
            r_res = item["RRT-Connect"]
            writer.writerow(
                [
                    item["seed"],
                    item.get("feasible", False),
                    a_res["success"],
                    f"{a_res['time_ms']:.6f}",
                    (f"{a_res['path_length']:.6f}" if a_res["success"] else ""),
                    a_res.get("nodes_expanded", 0),
                    r_res["success"],
                    f"{r_res['time_ms']:.6f}",
                    (f"{r_res['path_length']:.6f}" if r_res["success"] else ""),
                    r_res.get("vertices_sampled", 0),
                    r_res.get("strategy", "default"),
                ]
            )
    print(f"Task 2.3 CSV saved to {csv_path}")


def plot_phase2_paths(setup, mapper, path_results, student_id):
    plt.figure(figsize=(8, 8))
    plt.title(f"Phase 2 Paths in C-Space - ID: {student_id}")
    plt.imshow(mapper.cspace_grid.T, origin="lower", extent=setup["map_bounds"], cmap="Greys")

    sx, sy = setup["start"]
    gx, gy = setup["goal"]
    plt.plot(sx, sy, "go", markersize=9, label="Start")
    plt.plot(gx, gy, "r*", markersize=13, label="Goal")

    colors = {
        "A* (w=1.0)": "tab:blue",
        "Weighted A* (w=1.5)": "tab:orange",
        "Weighted A* (w=5.0)": "tab:purple",
        "RRT-Connect": "tab:green",
    }

    for name, res in path_results.items():
        if not res["success"] or not res["path"]:
            continue
        pts = np.array(res["path"])
        plt.plot(pts[:, 0], pts[:, 1], "-", linewidth=1.7, color=colors.get(name, "tab:red"), label=name)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("phase2_paths.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    print("Phase 2 plot saved to phase2_paths.png")


def choose_execution_path(phase2_results):
    """Pick the best available path to execute in simulation."""
    preference = [
        "A* (w=1.0)",
        "Weighted A* (w=1.5)",
        "Weighted A* (w=5.0)",
        "RRT-Connect",
    ]
    for name in preference:
        res = phase2_results.get(name)
        if res and res.get("success") and res.get("path"):
            return name, res["path"]
    return None, None


def wait_for_start_signal(auto_start_seconds=3):
    """Wait for Enter briefly, then auto-start to avoid blocking forever."""
    prompt = (
        f"Press Enter to activate dynamic obstacle and start simulation "
        f"(auto-start in {auto_start_seconds}s if input is unavailable)..."
    )
    print(prompt)
    if sys.stdin is None or not sys.stdin.isatty():
        time.sleep(auto_start_seconds)
        return

    try:
        ready, _, _ = select.select([sys.stdin], [], [], auto_start_seconds)
        if ready:
            # Consume the Enter press.
            sys.stdin.readline()
        else:
            print("No Enter detected. Auto-starting simulation.")
    except (OSError, ValueError):
        time.sleep(auto_start_seconds)


def execute_path_in_simulation(
    env,
    path,
    speed_mps=2.0,
    sim_hz=240,
    max_runtime_seconds=120.0,
    goal_hold_seconds=2.0,
):
    """
    Move the robot body along a planned path while stepping PyBullet.
    This gives a visible "working simulation" instead of a static scene.
    """
    if not path:
        raise ValueError("Path is empty; cannot execute simulation.")

    p = env_factory.p
    dt = 1.0 / sim_hz
    step_dist = speed_mps * dt
    robot_z = 0.1

    current = np.array(path[0], dtype=float)
    idx = 1

    # Place robot at the first waypoint.
    p.resetBasePositionAndOrientation(env.robot_id, [current[0], current[1], robot_z], [0, 0, 0, 1])

    t0 = time.perf_counter()
    while idx < len(path):
        if time.perf_counter() - t0 > max_runtime_seconds:
            print("Simulation timeout reached before finishing the path.")
            break

        target = np.array(path[idx], dtype=float)
        delta = target - current
        dist = np.linalg.norm(delta)

        if dist < 1e-9:
            idx += 1
            continue

        direction = delta / dist
        travel = min(step_dist, dist)
        current += direction * travel

        yaw = math.atan2(direction[1], direction[0])
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(env.robot_id, [current[0], current[1], robot_z], quat)

        env.update_simulation()
        time.sleep(dt)

        if dist <= step_dist + 1e-9:
            idx += 1

    # Keep sim alive briefly at the goal so movement completion is obvious.
    hold_steps = int(goal_hold_seconds * sim_hz)
    for _ in range(max(1, hold_steps)):
        env.update_simulation()
        time.sleep(dt)


def configure_interactive_viewer():
    """Hide side debug panes and enable interactive camera controls."""
    p = env_factory.p
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    if hasattr(p, "COV_ENABLE_MOUSE_PICKING"):
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)


def draw_path_debug_lines(path, color=(1, 0, 0), line_width=2.5, life_time=0):
    """Draw planned path in PyBullet so execution is visible."""
    p = env_factory.p
    if not path or len(path) < 2:
        return
    z = 0.05
    for i in range(1, len(path)):
        a = [float(path[i - 1][0]), float(path[i - 1][1]), z]
        b = [float(path[i][0]), float(path[i][1]), z]
        p.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=line_width, lifeTime=life_time)


def estimate_robot_radius(robot_type, robot_geometry):
    """Conservative radius of robot footprint around its center."""
    if robot_type == "RECT":
        half_w = robot_geometry[0] * 0.5
        half_l = robot_geometry[1] * 0.5
        return math.hypot(half_w, half_l)

    pts = np.array(robot_geometry, dtype=float)
    return float(np.max(np.linalg.norm(pts[:, :2], axis=1)))


def build_dynamic_obstacle_mask(mapper, dynamic_pos, inflated_radius):
    """Rasterize moving obstacle (inflated in C-space) into a boolean mask."""
    mask = np.zeros((mapper.width, mapper.height), dtype=bool)
    if dynamic_pos is None:
        return mask

    cx, cy = float(dynamic_pos[0]), float(dynamic_pos[1])
    gx, gy = mapper.world_to_grid(cx, cy)
    cell_radius = int(math.ceil(inflated_radius / mapper.res))
    r2 = inflated_radius * inflated_radius

    x_lo = max(0, gx - cell_radius)
    x_hi = min(mapper.width - 1, gx + cell_radius)
    y_lo = max(0, gy - cell_radius)
    y_hi = min(mapper.height - 1, gy + cell_radius)

    for ix in range(x_lo, x_hi + 1):
        for iy in range(y_lo, y_hi + 1):
            wx, wy = mapper.grid_to_world(ix, iy)
            if (wx - cx) ** 2 + (wy - cy) ** 2 <= r2:
                mask[ix, iy] = True
    return mask


def build_predictive_dynamic_mask(mapper, dynamic_positions, inflated_radius, future_radius_scale=0.55):
    """Current position as hard obstacle, future positions as softer risk zones."""
    mask = np.zeros((mapper.width, mapper.height), dtype=bool)
    if not dynamic_positions:
        return mask

    mask |= build_dynamic_obstacle_mask(mapper, dynamic_positions[0], inflated_radius)
    future_radius = max(mapper.res * 1.5, inflated_radius * future_radius_scale)
    for pos in dynamic_positions[1:]:
        mask |= build_dynamic_obstacle_mask(mapper, pos, future_radius)
    return mask


def crop_local_occupancy(
    full_occ,
    mapper,
    start_cell,
    goal_cell,
    predicted_positions,
    padding_cells=22,
    min_span_cells=80,
):
    """
    Build a local cropped occupancy around start/goal/predicted obstacle points.
    Returns (occ_crop, start_local, goal_local, (x0, y0)).
    """
    w, h = full_occ.shape
    sx, sy = int(start_cell[0]), int(start_cell[1])
    gx, gy = int(goal_cell[0]), int(goal_cell[1])

    xs = [sx, gx]
    ys = [sy, gy]
    for px, py in predicted_positions:
        cx, cy = mapper.world_to_grid(float(px), float(py))
        xs.append(int(np.clip(cx, 0, w - 1)))
        ys.append(int(np.clip(cy, 0, h - 1)))

    x0 = max(0, min(xs) - padding_cells)
    x1 = min(w - 1, max(xs) + padding_cells)
    y0 = max(0, min(ys) - padding_cells)
    y1 = min(h - 1, max(ys) + padding_cells)

    # Ensure minimum crop size for robust local planning.
    if (x1 - x0 + 1) < min_span_cells:
        miss = min_span_cells - (x1 - x0 + 1)
        left = miss // 2
        right = miss - left
        x0 = max(0, x0 - left)
        x1 = min(w - 1, x1 + right)
    if (y1 - y0 + 1) < min_span_cells:
        miss = min_span_cells - (y1 - y0 + 1)
        down = miss // 2
        up = miss - down
        y0 = max(0, y0 - down)
        y1 = min(h - 1, y1 + up)

    # Clamp exactly if still short due to boundaries.
    if (x1 - x0 + 1) < min_span_cells:
        if x0 == 0:
            x1 = min(w - 1, min_span_cells - 1)
        else:
            x0 = max(0, w - min_span_cells)
    if (y1 - y0 + 1) < min_span_cells:
        if y0 == 0:
            y1 = min(h - 1, min_span_cells - 1)
        else:
            y0 = max(0, h - min_span_cells)

    occ_crop = full_occ[x0 : x1 + 1, y0 : y1 + 1]
    start_local = (sx - x0, sy - y0)
    goal_local = (gx - x0, gy - y0)
    return occ_crop, start_local, goal_local, (x0, y0)


def predict_dynamic_positions(env, setup, sim_hz, horizon_s=1.2, sample_dt_s=0.2):
    """
    Predict moving obstacle positions for a short horizon.
    Uses env.dynamic_agent internal state (t, direction) when available.
    """
    agent = getattr(env, "dynamic_agent", None)
    if agent is None:
        return [tuple(setup["dynamic_obstacle"]["path_start"])]

    start = np.array(agent.start_node[:2], dtype=float)
    end = np.array(agent.end_node[:2], dtype=float)
    t = float(getattr(agent, "t", 0.0))
    forward = bool(getattr(agent, "forward", True))
    speed_t = float(getattr(agent, "speed", setup["dynamic_obstacle"]["speed"]))

    sample_steps = max(1, int(round(sample_dt_s * sim_hz)))
    samples = max(1, int(round(horizon_s / sample_dt_s)))

    positions = []
    for _ in range(samples + 1):
        pos = (1.0 - t) * start + t * end
        positions.append((float(pos[0]), float(pos[1])))

        for _ in range(sample_steps):
            t += speed_t if forward else -speed_t
            if t >= 1.0:
                t = 1.0
                forward = False
            elif t <= 0.0:
                t = 0.0
                forward = True

    return positions


def shortcut_path_by_los(path, mapper, max_jump=80):
    """String-pulling shortcut to remove zig-zag waypoints before smoothing."""
    if path is None or len(path) < 3:
        return path

    out = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        best = i + 1
        upper = min(n - 1, i + max_jump)
        for j in range(upper, i + 1, -1):
            if line_of_sight_collision_free(path[i], path[j], mapper):
                best = j
                break
        out.append(path[best])
        i = best
    return out


def validate_path_collision_free(path, mapper, check_segments=True):
    """Return True if sampled path points are free (and optionally segments too)."""
    if not path:
        return False
    for x, y in path:
        if not mapper.is_collision_free(x, y):
            return False
    if check_segments:
        for i in range(1, len(path)):
            if not line_of_sight_collision_free(path[i - 1], path[i], mapper):
                return False
    return True


def smooth_path_cubic_spline(raw_path, mapper, points_per_meter=8.0):
    """
    Task 3.1: Cubic spline smoothing (C2 where spline is valid).
    Falls back to raw path if smoothing becomes invalid/colliding.
    """
    if raw_path is None or len(raw_path) < 4:
        return raw_path, "raw"

    # Restore previous smoothing behavior: shortcut first, then natural cubic spline.
    raw_path = shortcut_path_by_los(raw_path, mapper, max_jump=80)
    pts = np.array(raw_path, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-8))
    pts = pts[keep]
    if len(pts) < 4:
        return [(float(p[0]), float(p[1])) for p in pts], "raw"

    t = np.zeros(len(pts), dtype=float)
    t[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    total = float(t[-1])
    if total < 1e-8:
        return [(float(p[0]), float(p[1])) for p in pts], "raw"

    sample_count = max(len(pts) * 2, int(math.ceil(total * max(points_per_meter, 10.0))))
    t_dense = np.linspace(0.0, total, sample_count)

    try:
        from scipy.interpolate import CubicSpline

        sx = CubicSpline(t, pts[:, 0], bc_type="natural")
        sy = CubicSpline(t, pts[:, 1], bc_type="natural")
        smoothed = np.column_stack((sx(t_dense), sy(t_dense)))
        method = "cubic_spline"
    except Exception:
        # Safe fallback if SciPy is unavailable.
        x_lin = np.interp(t_dense, t, pts[:, 0])
        y_lin = np.interp(t_dense, t, pts[:, 1])
        smoothed = np.column_stack((x_lin, y_lin))
        method = "linear_fallback"

    smoothed[0] = pts[0]
    smoothed[-1] = pts[-1]
    smooth_path = [(float(p[0]), float(p[1])) for p in smoothed]

    if np.all(np.isfinite(smoothed)):
        return smooth_path, method

    gradient = smooth_path_gradient_descent(raw_path, mapper)
    if gradient is not None and len(gradient) >= 2:
        return gradient, "gradient_fallback"

    return [(float(p[0]), float(p[1])) for p in pts], "raw_fallback"


def smooth_path_gradient_descent(raw_path, mapper, alpha_data=0.12, alpha_smooth=0.22, iterations=350):
    """Gradient-based path smoother that stays close to free-space raw path."""
    if raw_path is None or len(raw_path) < 3:
        return raw_path

    original = np.array(raw_path, dtype=float)
    smooth = original.copy()
    n = len(smooth)

    for _ in range(iterations):
        for i in range(1, n - 1):
            candidate = (
                smooth[i]
                + alpha_data * (original[i] - smooth[i])
                + alpha_smooth * (smooth[i - 1] + smooth[i + 1] - 2.0 * smooth[i])
            )
            if mapper.is_collision_free(candidate[0], candidate[1]):
                smooth[i] = candidate

    # Snap any invalid samples back to raw path samples.
    for i in range(1, n - 1):
        if not mapper.is_collision_free(smooth[i][0], smooth[i][1]):
            smooth[i] = original[i]

    return [(float(p[0]), float(p[1])) for p in smooth]


def smooth_replanned_path_online(path, mapper, passes=3):
    """Lightweight smoother for each newly replanned local path."""
    if path is None or len(path) < 3:
        return path

    arr = np.array(path, dtype=float)
    n = len(arr)
    for _ in range(passes):
        for i in range(1, n - 1):
            candidate = 0.2 * arr[i - 1] + 0.6 * arr[i] + 0.2 * arr[i + 1]
            if (
                mapper.is_collision_free(candidate[0], candidate[1])
                and line_of_sight_collision_free(arr[i - 1], candidate, mapper)
                and line_of_sight_collision_free(candidate, arr[i + 1], mapper)
            ):
                arr[i] = candidate
    return [(float(p[0]), float(p[1])) for p in arr]


def choose_lookahead_target(robot_pos, waypoint_queue, lookahead_dist=0.9):
    """Choose a target ahead on the queue for smoother steering."""
    if len(waypoint_queue) == 0:
        return None

    prev = np.array(robot_pos, dtype=float)
    acc = 0.0
    for wp in waypoint_queue:
        seg = np.linalg.norm(wp - prev)
        if acc + seg >= lookahead_dist:
            ratio = (lookahead_dist - acc) / max(seg, 1e-9)
            return prev + ratio * (wp - prev)
        acc += seg
        prev = wp

    return waypoint_queue[-1].copy()


def path_intersects_predicted_obstacle(waypoint_queue, predicted_positions, clearance, max_waypoints=28):
    """Check whether near-future waypoints will enter predicted obstacle region."""
    if len(waypoint_queue) == 0 or not predicted_positions:
        return False

    c2 = clearance * clearance
    count = 0
    for wp in waypoint_queue:
        for pos in predicted_positions:
            dx = float(wp[0]) - float(pos[0])
            dy = float(wp[1]) - float(pos[1])
            if dx * dx + dy * dy <= c2:
                return True
        count += 1
        if count >= max_waypoints:
            break
    return False


def nearest_path_index(path, pos, start_idx=0, window=120):
    """Find nearest waypoint index in a forward window."""
    if not path:
        return 0
    lo = max(0, min(start_idx, len(path) - 1))
    hi = min(len(path), lo + max(1, window))
    best_i = lo
    best_d2 = float("inf")
    px, py = float(pos[0]), float(pos[1])
    for i in range(lo, hi):
        dx = float(path[i][0]) - px
        dy = float(path[i][1]) - py
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def path_conflict_index(path, start_idx, predicted_positions, clearance, lookahead_count=32):
    """Return first conflicting waypoint index; None if no conflict."""
    if not path or not predicted_positions:
        return None
    c2 = clearance * clearance
    end = min(len(path), start_idx + max(1, lookahead_count))
    for i in range(start_idx, end):
        wx, wy = float(path[i][0]), float(path[i][1])
        for px, py in predicted_positions:
            dx = wx - float(px)
            dy = wy - float(py)
            if dx * dx + dy * dy <= c2:
                return i
    return None


def find_rejoin_index(base_path, from_idx, predicted_positions, clearance, min_offset=14, max_offset=52):
    """Pick a rejoin waypoint on original path after the conflict zone."""
    if not base_path:
        return None

    start = min(len(base_path) - 1, from_idx + max(1, min_offset))
    end = min(len(base_path) - 1, from_idx + max(min_offset + 1, max_offset))
    if start > end:
        return None

    c2 = clearance * clearance
    for i in range(start, end + 1):
        wx, wy = float(base_path[i][0]), float(base_path[i][1])
        safe = True
        for px, py in predicted_positions:
            dx = wx - float(px)
            dy = wy - float(py)
            if dx * dx + dy * dy <= c2:
                safe = False
                break
        if safe:
            return i
    return None


def plot_phase3_replanning(
    setup,
    mapper,
    replanned_paths,
    executed_trail,
    student_id,
    reference_path=None,
    replan_events=None,
):
    """Save an explanatory Task 3 figure: original plan, dynamic corridor, and replans."""
    plt.figure(figsize=(9, 9))
    plt.title(f"Task 3 D* Lite Replanning & Executed Trail - ID: {student_id}")
    plt.imshow(mapper.cspace_grid.T, origin="lower", extent=setup["map_bounds"], cmap="Greys")

    # Show original route before dynamic replanning.
    if reference_path and len(reference_path) >= 2:
        ref = np.array(reference_path)
        plt.plot(
            ref[:, 0],
            ref[:, 1],
            "--",
            color=(1.0, 0.55, 0.1, 0.95),
            linewidth=2.0,
            label="Original Planned Path",
        )

    # Visualize dynamic obstacle swept corridor (inflated by robot radius in C-space).
    dyn = setup.get("dynamic_obstacle", {})
    if "path_start" in dyn and "path_end" in dyn and "radius" in dyn:
        p0 = np.array(dyn["path_start"], dtype=float)
        p1 = np.array(dyn["path_end"], dtype=float)
        v = p1 - p0
        vn = np.linalg.norm(v)
        if vn > 1e-9:
            u = v / vn
            n = np.array([-u[1], u[0]])
            inflated = dyn["radius"] + estimate_robot_radius(setup["robot_type"], setup["robot_geometry"]) + mapper.res
            poly = np.vstack([p0 + n * inflated, p1 + n * inflated, p1 - n * inflated, p0 - n * inflated])
            corridor = MplPolygon(
                poly,
                closed=True,
                facecolor=(0.2, 0.55, 1.0, 0.18),
                edgecolor=(0.1, 0.45, 0.95, 0.6),
                linewidth=1.2,
                label="Dynamic Obstacle Swept Zone",
            )
            plt.gca().add_patch(corridor)
            plt.plot([p0[0], p1[0]], [p0[1], p1[1]], ":", color=(0.1, 0.35, 0.9, 0.95), linewidth=1.6, label="Obstacle Centerline")

    # Show sampled replanned alternatives (faint cyan).
    if replanned_paths:
        stride = max(1, len(replanned_paths) // 35)
        for i, path in enumerate(replanned_paths):
            if i % stride != 0:
                continue
            pts = np.array(path)
            if len(pts) >= 2:
                plt.plot(pts[:, 0], pts[:, 1], color=(0.0, 0.75, 1.0, 0.22), linewidth=1.2)

    # Mark where replanning happened and where it rejoins original path.
    if replan_events:
        for i, evt in enumerate(replan_events):
            rp = evt.get("robot_pos")
            cp = evt.get("conflict_point")
            jp = evt.get("rejoin_point")
            if cp is not None:
                plt.plot(
                    cp[0],
                    cp[1],
                    "x",
                    color=(0.95, 0.25, 0.25, 1.0),
                    markersize=9,
                    markeredgewidth=2.0,
                    label="Predicted Conflict" if i == 0 else None,
                )
            if jp is not None:
                plt.plot(
                    jp[0],
                    jp[1],
                    "^",
                    color=(0.1, 0.75, 0.9, 1.0),
                    markersize=8,
                    label="Rejoin Point" if i == 0 else None,
                )
            if rp is not None:
                plt.plot(
                    rp[0],
                    rp[1],
                    "o",
                    color=(0.5, 0.0, 0.8, 0.85),
                    markersize=5,
                    label="Replan Trigger" if i == 0 else None,
                )

    if executed_trail and len(executed_trail) >= 2:
        tr = np.array(executed_trail)
        plt.plot(tr[:, 0], tr[:, 1], color=(1.0, 0.2, 0.8, 0.95), linewidth=2.8, label="Executed Trail")

    sx, sy = setup["start"]
    gx, gy = setup["goal"]
    plt.plot(sx, sy, "go", markersize=9, label="Start")
    plt.plot(gx, gy, "r*", markersize=13, label="Goal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("phase3_dstar_replanning.png", dpi=180)
    plt.close()
    print("Phase 3 replanning plot saved to phase3_dstar_replanning.png")


def plot_phase3_smoothing(setup, mapper, raw_path, smooth_path, student_id):
    """Plot Task 3.1 before/after trajectories on C-space."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sx, sy = setup["start"]
    gx, gy = setup["goal"]

    ax1 = axes[0]
    ax1.set_title("Task 3.1 Before (Raw)")
    ax1.imshow(mapper.cspace_grid.T, origin="lower", extent=setup["map_bounds"], cmap="Greys")
    if raw_path:
        raw = np.array(raw_path)
        ax1.plot(raw[:, 0], raw[:, 1], color="tab:red", linewidth=1.8, label="Raw")
    ax1.plot(sx, sy, "go", markersize=9, label="Start")
    ax1.plot(gx, gy, "r*", markersize=13, label="Goal")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.legend(loc="best")

    ax2 = axes[1]
    ax2.set_title("Task 3.1 After (Smoothed)")
    ax2.imshow(mapper.cspace_grid.T, origin="lower", extent=setup["map_bounds"], cmap="Greys")
    if smooth_path:
        sm = np.array(smooth_path)
        ax2.plot(sm[:, 0], sm[:, 1], color="tab:green", linewidth=2.0, label="Smoothed")
    ax2.plot(sx, sy, "go", markersize=9, label="Start")
    ax2.plot(gx, gy, "r*", markersize=13, label="Goal")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig("phase3_smoothing_before_after.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    print("Phase 3 smoothing plot saved to phase3_smoothing_before_after.png")


class DStarLitePlanner:
    """Task 3.2: D* Lite on an 8-connected occupancy grid."""

    def __init__(self, occupancy, start, goal, resolution):
        self.occupancy = occupancy.astype(bool).copy()
        self.width, self.height = self.occupancy.shape
        self.resolution = resolution
        self.start = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))

        self.km = 0.0
        self.g = np.full((self.width, self.height), np.inf, dtype=float)
        self.rhs = np.full((self.width, self.height), np.inf, dtype=float)
        self.rhs[self.goal[0], self.goal[1]] = 0.0

        self.open_heap = []
        self.open_dict = {}
        self.total_expansions = 0

        self._push(self.goal, self._calculate_key(self.goal))

    def _heuristic(self, a, b):
        return octile_distance(a, b) * self.resolution

    def _in_bounds(self, n):
        return 0 <= n[0] < self.width and 0 <= n[1] < self.height

    def _is_blocked(self, n):
        return self.occupancy[n[0], n[1]]

    def _neighbors(self, n, include_blocked=False):
        out = []
        x, y = n
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if not self._in_bounds(nb):
                    continue
                if include_blocked or not self._is_blocked(nb):
                    out.append(nb)
        return out

    def _edge_cost(self, a, b):
        if not self._in_bounds(a) or not self._in_bounds(b):
            return np.inf
        if self._is_blocked(a) or self._is_blocked(b):
            return np.inf
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if dx + dy == 1:
            return self.resolution
        if dx == 1 and dy == 1:
            return self.resolution * math.sqrt(2)
        return np.inf

    def _calculate_key(self, s):
        g_rhs = min(self.g[s[0], s[1]], self.rhs[s[0], s[1]])
        return (g_rhs + self._heuristic(self.start, s) + self.km, g_rhs)

    @staticmethod
    def _key_less(a, b):
        eps = 1e-9
        if a[0] < b[0] - eps:
            return True
        if b[0] < a[0] - eps:
            return False
        return a[1] < b[1] - eps

    def _push(self, node, key):
        self.open_dict[node] = key
        heapq.heappush(self.open_heap, (key[0], key[1], node[0], node[1]))

    def _remove(self, node):
        if node in self.open_dict:
            del self.open_dict[node]

    def _entry_matches(self, node, key):
        live = self.open_dict.get(node)
        if live is None:
            return False
        return abs(live[0] - key[0]) <= 1e-9 and abs(live[1] - key[1]) <= 1e-9

    def _top_key(self):
        while self.open_heap:
            k1, k2, x, y = self.open_heap[0]
            node = (x, y)
            if self._entry_matches(node, (k1, k2)):
                return (k1, k2)
            heapq.heappop(self.open_heap)
        return (np.inf, np.inf)

    def _pop_valid(self):
        while self.open_heap:
            k1, k2, x, y = heapq.heappop(self.open_heap)
            node = (x, y)
            if self._entry_matches(node, (k1, k2)):
                del self.open_dict[node]
                return node, (k1, k2)
        return None, (np.inf, np.inf)

    def update_vertex(self, u):
        ux, uy = u
        if u != self.goal:
            best = np.inf
            if not self._is_blocked(u):
                for s in self._neighbors(u, include_blocked=False):
                    cost = self._edge_cost(u, s) + self.g[s[0], s[1]]
                    if cost < best:
                        best = cost
            self.rhs[ux, uy] = best

        self._remove(u)
        if not math.isclose(self.g[ux, uy], self.rhs[ux, uy], rel_tol=0.0, abs_tol=1e-9):
            self._push(u, self._calculate_key(u))

    def move_start(self, new_start):
        new_start = (int(new_start[0]), int(new_start[1]))
        if new_start == self.start:
            return
        self.km += self._heuristic(self.start, new_start)
        self.start = new_start

    def apply_environment_changes(self, new_occupancy):
        new_occupancy = new_occupancy.astype(bool)
        changed = np.argwhere(new_occupancy != self.occupancy)
        if changed.size == 0:
            return 0

        self.occupancy = new_occupancy
        impacted = set()
        for cx, cy in changed:
            node = (int(cx), int(cy))
            impacted.add(node)
            for nb in self._neighbors(node, include_blocked=True):
                impacted.add(nb)

        for node in impacted:
            self.update_vertex(node)

        return int(changed.shape[0])

    def compute_shortest_path(self, max_steps=250000):
        steps = 0
        while (
            self._key_less(self._top_key(), self._calculate_key(self.start))
            or not math.isclose(
                self.rhs[self.start[0], self.start[1]],
                self.g[self.start[0], self.start[1]],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            if steps >= max_steps:
                break

            u, k_old = self._pop_valid()
            if u is None:
                break

            k_new = self._calculate_key(u)
            ux, uy = u

            if self._key_less(k_old, k_new):
                self._push(u, k_new)
            elif self.g[ux, uy] > self.rhs[ux, uy]:
                self.g[ux, uy] = self.rhs[ux, uy]
                for p in self._neighbors(u, include_blocked=True):
                    self.update_vertex(p)
            else:
                self.g[ux, uy] = np.inf
                self.update_vertex(u)
                for p in self._neighbors(u, include_blocked=True):
                    self.update_vertex(p)

            steps += 1

        self.total_expansions += steps
        return steps

    def extract_path(self, max_nodes=6000):
        if self._is_blocked(self.start):
            return None
        if not np.isfinite(self.g[self.start[0], self.start[1]]) and not np.isfinite(self.rhs[self.start[0], self.start[1]]):
            return None

        path = [self.start]
        cur = self.start
        visited = {cur}

        for _ in range(max_nodes):
            if cur == self.goal:
                return path

            best_nb = None
            best_score = np.inf
            for nb in self._neighbors(cur, include_blocked=False):
                score = self._edge_cost(cur, nb) + self.g[nb[0], nb[1]]
                if score < best_score - 1e-9:
                    best_score = score
                    best_nb = nb

            if best_nb is None or not np.isfinite(best_score):
                return None
            if best_nb in visited:
                return None

            path.append(best_nb)
            visited.add(best_nb)
            cur = best_nb

        return None


def run_dstar_lite_dynamic_simulation(
    env,
    mapper,
    setup,
    reference_path=None,
    sim_hz=240,
    replan_hz=4,
    robot_speed_mps=1.5,
    max_runtime_seconds=120.0,
    prediction_horizon_s=2.5,
    prediction_dt_s=0.2,
    visualize_live_plan=False,
    live_plan_draw_hz=2,
    show_robot_trail=True,
    local_replan_ahead=32,
    rejoin_min_offset=8,
    rejoin_max_offset=28,
    local_plan_max_steps=35000,
    local_crop_padding_cells=22,
    local_crop_min_span_cells=80,
    sleep_in_loop=True,
):
    """
    Task 3.2 + 3.3:
    Run local-segment D* Lite replanning:
    only replan affected segment when predicted obstacle conflicts with the path.
    """
    p = env_factory.p
    dt = 1.0 / sim_hz
    replan_period = max(1, int(round(sim_hz / replan_hz)))
    draw_period = max(1, int(round(sim_hz / live_plan_draw_hz)))
    step_dist = robot_speed_mps * dt
    robot_z = 0.1

    static_occ = mapper.cspace_grid.astype(bool)
    robot_radius = estimate_robot_radius(setup["robot_type"], setup["robot_geometry"])
    dyn_radius = setup["dynamic_obstacle"]["radius"] + robot_radius + mapper.res

    goal_world = np.array(setup["goal"], dtype=float)

    dyn_pos = setup["dynamic_obstacle"]["path_start"]

    robot_pos = np.array(setup["start"], dtype=float)
    p.resetBasePositionAndOrientation(env.robot_id, [robot_pos[0], robot_pos[1], robot_z], [0, 0, 0, 1])

    if reference_path is None or len(reference_path) < 2:
        # Fallback: one-shot global D* plan if no reference path provided.
        start_cell = mapper.world_to_grid(setup["start"][0], setup["start"][1])
        goal_cell = mapper.world_to_grid(setup["goal"][0], setup["goal"][1])
        occ0 = np.logical_or(static_occ, build_dynamic_obstacle_mask(mapper, dyn_pos, dyn_radius))
        fallback_planner = DStarLitePlanner(occ0, start_cell, goal_cell, mapper.res)
        fallback_planner.compute_shortest_path()
        grid_path = fallback_planner.extract_path()
        if grid_path is None or len(grid_path) < 2:
            return {
                "success": False,
                "final_distance_to_goal": float(np.linalg.norm(robot_pos - goal_world)),
                "replans_due_to_changes": 0,
                "replan_updates": 0,
                "waypoint_switches": 0,
                "planning_time_ms": 0.0,
                "dstar_expansions": fallback_planner.total_expansions,
                "replanned_paths_history": [],
                "executed_trail": [(float(robot_pos[0]), float(robot_pos[1]))],
                "replan_events": [],
            }
        base_path = [mapper.grid_to_world(ix, iy) for ix, iy in grid_path]
    else:
        base_path = [(float(p[0]), float(p[1])) for p in reference_path]

    active_path = list(base_path)
    active_idx = 1
    waypoint_switches = 0
    replans = 0
    replan_updates = 0
    planning_time_ms = 0.0
    dstar_expansions_total = 0
    replanned_paths_history = []
    executed_trail = [(float(robot_pos[0]), float(robot_pos[1]))]
    replan_events = []

    max_steps = int(max_runtime_seconds * sim_hz)
    reached_goal = False
    goal_tolerance = max(0.25, mapper.res * 1.5)
    last_local_replan_step = -10**9
    min_replan_interval_steps = max(1, int(round(0.8 * sim_hz)))

    for step in range(max_steps):
        loop_t0 = time.perf_counter()
        dyn_pos = env.update_simulation()
        if dyn_pos is None:
            dyn_pos = setup["dynamic_obstacle"]["path_start"]

        if step % replan_period == 0:
            replan_updates += 1

            active_idx = max(active_idx, nearest_path_index(active_path, robot_pos, max(0, active_idx - 4), window=80))
            base_idx = nearest_path_index(base_path, robot_pos, max(0, min(len(base_path) - 1, active_idx - 8)), window=120)
            predicted_positions = predict_dynamic_positions(
                env,
                setup,
                sim_hz,
                horizon_s=prediction_horizon_s,
                sample_dt_s=prediction_dt_s,
            )
            if not predicted_positions:
                predicted_positions = [tuple(dyn_pos)]
            conflict_idx = path_conflict_index(
                active_path,
                active_idx,
                predicted_positions,
                clearance=dyn_radius + mapper.res * 2.0,
                lookahead_count=local_replan_ahead,
            )

            need_replan = (
                conflict_idx is not None
                and active_idx < len(active_path) - 2
                and (step - last_local_replan_step) >= min_replan_interval_steps
            )

            if need_replan:
                # Enter cooldown as soon as we decide to attempt a local replan,
                # so repeated failed attempts do not freeze the simulation.
                last_local_replan_step = step

                conflict_pt = active_path[conflict_idx]
                conflict_base_idx = nearest_path_index(
                    base_path,
                    conflict_pt,
                    start_idx=base_idx,
                    window=180,
                )
                rejoin_idx = find_rejoin_index(
                    base_path,
                    conflict_base_idx,
                    predicted_positions,
                    clearance=dyn_radius + mapper.res * 1.0,
                    min_offset=rejoin_min_offset,
                    max_offset=rejoin_max_offset,
                )
                if rejoin_idx is None:
                    continue

                local_occ_full = np.logical_or(static_occ, build_dynamic_obstacle_mask(mapper, dyn_pos, dyn_radius))
                local_occ_full |= build_predictive_dynamic_mask(
                    mapper,
                    predicted_positions[:6],
                    dyn_radius,
                    future_radius_scale=0.45,
                )
                start_cell = mapper.world_to_grid(robot_pos[0], robot_pos[1])
                start_cell = (
                    int(np.clip(start_cell[0], 0, mapper.width - 1)),
                    int(np.clip(start_cell[1], 0, mapper.height - 1)),
                )
                goal_cell = mapper.world_to_grid(base_path[rejoin_idx][0], base_path[rejoin_idx][1])
                goal_cell = (
                    int(np.clip(goal_cell[0], 0, mapper.width - 1)),
                    int(np.clip(goal_cell[1], 0, mapper.height - 1)),
                )

                local_occ, start_local, goal_local, crop_offset = crop_local_occupancy(
                    local_occ_full,
                    mapper,
                    start_cell,
                    goal_cell,
                    predicted_positions[:6],
                    padding_cells=local_crop_padding_cells,
                    min_span_cells=local_crop_min_span_cells,
                )
                if local_occ[start_local[0], start_local[1]] or local_occ[goal_local[0], goal_local[1]]:
                    continue

                local_planner = DStarLitePlanner(local_occ, start_local, goal_local, mapper.res)
                t0 = time.perf_counter()
                local_planner.compute_shortest_path(max_steps=local_plan_max_steps)
                planning_time_ms += (time.perf_counter() - t0) * 1000.0
                dstar_expansions_total += local_planner.total_expansions

                local_grid_path = local_planner.extract_path(max_nodes=1600)
                if local_grid_path is not None and len(local_grid_path) > 1:
                    ox, oy = crop_offset
                    local_world = [mapper.grid_to_world(ix + ox, iy + oy) for ix, iy in local_grid_path]
                    local_world[0] = (float(robot_pos[0]), float(robot_pos[1]))
                    local_world = smooth_replanned_path_online(local_world, mapper, passes=1)

                    # Replan only affected segment: [current -> local_detour -> original_suffix]
                    new_path = [local_world[0]] + local_world[1:] + list(base_path[rejoin_idx + 1 :])
                    active_path = new_path
                    active_idx = 1
                    replans += 1
                    replan_events.append(
                        {
                            "robot_pos": (float(robot_pos[0]), float(robot_pos[1])),
                            "conflict_point": (
                                float(conflict_pt[0]),
                                float(conflict_pt[1]),
                            ),
                            "rejoin_point": (
                                float(base_path[rejoin_idx][0]),
                                float(base_path[rejoin_idx][1]),
                            ),
                        }
                    )

                    replanned_paths_history.append(list(active_path))
                    if len(replanned_paths_history) > 180:
                        replanned_paths_history.pop(0)

                    if visualize_live_plan and (step % draw_period == 0):
                        draw_path_debug_lines(
                            active_path,
                            color=(0.0, 0.9, 1.0),
                            line_width=2.0,
                            life_time=max(0.2, 2.0 / max(1e-9, live_plan_draw_hz)),
                        )

        if active_idx < len(active_path):
            while active_idx < len(active_path):
                wp = np.array(active_path[active_idx], dtype=float)
                if np.linalg.norm(wp - robot_pos) <= max(step_dist * 1.3, mapper.res * 0.6):
                    active_idx += 1
                    waypoint_switches += 1
                else:
                    break

        if active_idx < len(active_path):
            lookahead_slice = [
                np.array(pt, dtype=float)
                for pt in active_path[active_idx : min(len(active_path), active_idx + 24)]
            ]
            target = choose_lookahead_target(robot_pos, lookahead_slice, lookahead_dist=0.9)
            if target is None:
                target = np.array(active_path[active_idx], dtype=float)

            delta = target - robot_pos
            dist = np.linalg.norm(delta)

            if dist > 1e-9:
                prev_pos = robot_pos.copy()
                direction = delta / dist
                travel = min(step_dist, dist)
                robot_pos += direction * travel
                yaw = math.atan2(direction[1], direction[0])
                quat = p.getQuaternionFromEuler([0, 0, yaw])
                p.resetBasePositionAndOrientation(env.robot_id, [robot_pos[0], robot_pos[1], robot_z], quat)
                if show_robot_trail:
                    p.addUserDebugLine(
                        [float(prev_pos[0]), float(prev_pos[1]), 0.08],
                        [float(robot_pos[0]), float(robot_pos[1]), 0.08],
                        lineColorRGB=[1.0, 0.15, 0.85],
                        lineWidth=2.2,
                        lifeTime=0,
                    )
                if step % 3 == 0:
                    executed_trail.append((float(robot_pos[0]), float(robot_pos[1])))

        if np.linalg.norm(robot_pos - goal_world) <= goal_tolerance:
            reached_goal = True
            break

        if sleep_in_loop:
            elapsed = time.perf_counter() - loop_t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    return {
        "success": reached_goal,
        "final_distance_to_goal": float(np.linalg.norm(robot_pos - goal_world)),
        "replans_due_to_changes": replans,
        "replan_updates": replan_updates,
        "waypoint_switches": waypoint_switches,
        "planning_time_ms": planning_time_ms,
        "dstar_expansions": dstar_expansions_total,
        "replanned_paths_history": replanned_paths_history,
        "executed_trail": executed_trail,
        "replan_events": replan_events,
    }
