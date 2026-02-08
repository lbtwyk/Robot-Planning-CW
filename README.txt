===========================================================
ROBOT PLANNING MODULE: COURSEWORK 1 STARTER KIT
===========================================================

1. OVERVIEW
This package contains the 'env_factory', a python FILE 
that generates a randomized warehouse environment for 
robot planning tasks. Your goal is to navigate a robot from 
a start to a goal while avoiding static and dynamic obstacles.

2. SYSTEM REQUIREMENTS
- Python 3.11 or higher 
- PyBullet
- NumPy
- Matplotlib (for C-space visualization)
- SciPy (Recommended for path smoothing)

3. FILE STRUCTURE
- env_factory.py    : The core environment file (Do not modify/rename)
- planner.py        : Planning implementations (A*, Weighted A*, RRT-Connect, D* Lite, smoothing, plotting)
- main.py           : Coursework entrypoint that runs the full pipeline end-to-end
- starter_code.py   : Compatibility wrapper that calls main.py

4. IMPORTANT NOTES
- Initialisation: You MUST use the last 4 digits your Student ID as the seed in 
  the RandomizedWarehouse constructor.
- Verification: Your 'version_hash' is unique to your ID. Any 
  modifications to the environment logic will result in a 
  hash mismatch during grading.
- Coordination: The dynamic obstacle is only visible after 
  calling 'env.activate_dynamic_obstacle()'.

===========================================================
