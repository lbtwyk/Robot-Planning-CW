import argparse
import time

import matplotlib.pyplot as plt

import env_factory
import planner


def run_navigation(student_id=2557, gui=True, keep_alive=True):
    """Run the full coursework pipeline from map setup to dynamic replanning."""
    mode = env_factory.p.GUI if gui else env_factory.p.DIRECT
    env = env_factory.RandomizedWarehouse(seed=student_id, mode=mode)
    if gui:
        planner.configure_interactive_viewer()

    setup = env.get_problem_setup()
    print(f"Loaded environment for ID: {student_id}")
    print(f"Robot Geometry: {setup['robot_geometry']}")

    mapper = planner.GridMapper(setup, resolution=0.1)
    mapper.fill_obstacles(setup["static_obstacles"])

    # Phase 0: workspace snapshot with robot at start.
    start_x, start_y = setup["start"]
    vis_grid = mapper.overlay_robot(start_x, start_y, setup["robot_type"], setup["robot_geometry"])
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_grid.T, origin="lower", extent=setup["map_bounds"], cmap="viridis")
    plt.colorbar(label="0: Empty, 1: Obstacle, 2: Robot")
    plt.title(f"Warehouse Map with Robot at Start - ID: {student_id}")
    plt.savefig("phase0_workspace_with_robot.png", dpi=150)
    if planner.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # Phase 1: C-space generation and visualization.
    cspace_grid = mapper.compute_cspace(
        setup["static_obstacles"],
        setup["robot_type"],
        setup["robot_geometry"],
    )

    sx, sy = setup["start"]
    gx, gy = setup["goal"]
    print("Task 1.3 note: In general, C-space dimension equals robot DOF.")
    print("This coursework uses planar translation only, so C-space here is 2D: q=(x, y).")
    print(f"Start ({sx:.1f}, {sy:.1f}) collision-free: {mapper.is_collision_free(sx, sy)}")
    print(f"Goal  ({gx:.1f}, {gy:.1f}) collision-free: {mapper.is_collision_free(gx, gy)}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1 = axes[0]
    ax1.set_title(f"Workspace W (Euclidean plane, x-y) - ID: {student_id}")
    ax1.imshow(vis_grid.T, origin="lower", extent=setup["map_bounds"], cmap="viridis")
    ax1.plot(sx, sy, "go", markersize=10, label="Start")
    ax1.plot(gx, gy, "r*", markersize=15, label="Goal")
    ax1.legend()
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")

    ax2 = axes[1]
    ax2.set_title(f"C-Space C (Configuration q = [x, y]) - ID: {student_id}")
    ax2.imshow(cspace_grid.T, origin="lower", extent=setup["map_bounds"], cmap="Greys")
    ax2.plot(sx, sy, "go", markersize=10, label="Start")
    ax2.plot(gx, gy, "r*", markersize=15, label="Goal")
    ax2.legend()
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    plt.tight_layout()
    plt.savefig("phase1_workspace_vs_cspace.png", dpi=150)
    if planner.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    print("Phase 1 plot saved to phase1_workspace_vs_cspace.png")

    # Phase 2: Static planning comparison.
    a_star_result = planner.astar_search(mapper, setup["start"], setup["goal"], weight=1.0)
    wa_star_15_result = planner.astar_search(mapper, setup["start"], setup["goal"], weight=1.5)
    wa_star_50_result = planner.astar_search(mapper, setup["start"], setup["goal"], weight=5.0)
    rrt_result = planner.rrt_connect_planner(
        mapper=mapper,
        start_xy=setup["start"],
        goal_xy=setup["goal"],
        bounds=setup["map_bounds"],
        step_size=0.5,
        max_iterations=15000,
        goal_bias=0.1,
        rng_seed=student_id,
    )

    phase2_results = {
        "A* (w=1.0)": a_star_result,
        "Weighted A* (w=1.5)": wa_star_15_result,
        "Weighted A* (w=5.0)": wa_star_50_result,
        "RRT-Connect": rrt_result,
    }

    planner.print_phase2_comparison_table(phase2_results)
    planner.plot_phase2_paths(setup, mapper, phase2_results, student_id)

    # Phase 3: smoothing + dynamic replanning.
    _, execution_path = planner.choose_execution_path(phase2_results)
    if execution_path is None:
        print("No valid path found in Phase 2. Skipping Phase 3 execution.")
        return

    smooth_path, smooth_mode = planner.smooth_path_cubic_spline(execution_path, mapper)
    print(
        f"Task 3.1 smoothing mode: {smooth_mode} | "
        f"raw waypoints={len(execution_path)} -> smoothed waypoints={len(smooth_path)}"
    )
    planner.plot_phase3_smoothing(setup, mapper, execution_path, smooth_path, student_id)

    if gui:
        planner.wait_for_start_signal(auto_start_seconds=3)
    env.activate_dynamic_obstacle()
    print("Task 3.2/3.3: Running D* Lite with dynamic obstacle active.")

    if gui:
        planner.draw_path_debug_lines(execution_path, color=(1.0, 0.2, 0.2), line_width=2.0, life_time=0)
        planner.draw_path_debug_lines(smooth_path, color=(0.2, 1.0, 0.2), line_width=2.5, life_time=0)
        env_factory.p.resetDebugVisualizerCamera(
            cameraDistance=24.0,
            cameraYaw=45.0,
            cameraPitch=-65.0,
            cameraTargetPosition=[0.0, 0.0, 0.0],
        )

    result = planner.run_dstar_lite_dynamic_simulation(
        env=env,
        mapper=mapper,
        setup=setup,
        reference_path=smooth_path,
        sim_hz=240,
        replan_hz=4,
        robot_speed_mps=1.5,
        max_runtime_seconds=120.0,
        prediction_horizon_s=2.5,
        prediction_dt_s=0.2,
        visualize_live_plan=False,
        live_plan_draw_hz=2,
        show_robot_trail=gui,
        local_replan_ahead=32,
        rejoin_min_offset=8,
        rejoin_max_offset=28,
        sleep_in_loop=gui,
    )

    print("\n=== Phase 3 D* Lite Summary ===")
    print(f"Success: {result['success']}")
    print(f"Final distance to goal (m): {result['final_distance_to_goal']:.3f}")
    print(f"Re-plans due to map changes: {result['replans_due_to_changes']}")
    print(f"Replan updates: {result['replan_updates']}")
    print(f"Waypoint switches: {result['waypoint_switches']}")
    print(f"Planning time total (ms): {result['planning_time_ms']:.2f}")
    print(f"D* Lite expansions: {result['dstar_expansions']}")

    planner.plot_phase3_replanning(
        setup=setup,
        mapper=mapper,
        replanned_paths=result["replanned_paths_history"],
        executed_trail=result["executed_trail"],
        student_id=student_id,
        reference_path=smooth_path,
        replan_events=result["replan_events"],
    )

    if gui and keep_alive:
        try:
            while True:
                env.update_simulation()
                time.sleep(1.0 / 240.0)
        except KeyboardInterrupt:
            print("Simulation ended.")


def main():
    parser = argparse.ArgumentParser(description="Robot planning coursework entrypoint")
    parser.add_argument("--student-id", type=int, default=2557, help="Last 4 digits of student ID")
    parser.add_argument("--headless", action="store_true", help="Run in DIRECT mode without GUI")
    parser.add_argument("--no-hold", action="store_true", help="Exit after planning instead of keeping GUI open")
    args = parser.parse_args()

    run_navigation(student_id=args.student_id, gui=not args.headless, keep_alive=not args.no_hold)


if __name__ == "__main__":
    main()
