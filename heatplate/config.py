"""Problem data for ME5704 Project 2 (all values taken from the assignment sheet)."""
import numpy as np

# --- geometry -----------------------------------------------------------------
L = 4.0                     # plate is 4 x 4, unit thickness
THICKNESS = 1.0

# --- boundary / source temperatures (Dirichlet) -------------------------------
T_BOUNDARY = 200.0          # [K] all four outer edges
T_SOURCE = 350.0            # [K] heat source along the red line
SOURCE_X = 1.5              # red line: x = 1.5,
SOURCE_Y = (2.0, 3.0)       #           2 <= y <= 3

# --- the 9 interior points asked for in part (a) -------------------------------
POINTS_9 = [(1, 1), (2, 1), (3, 1),
            (1, 2), (2, 2), (3, 2),
            (1, 3), (2, 3), (3, 3)]          # (x, y) coordinates

# --- thermal-conductivity data for part (c) ------------------------------------
T_DATA = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
K_DATA = np.array([0.10, 0.25, 0.35, 0.40, 0.42])

# --- numerical settings ---------------------------------------------------------
GRID_SPACINGS = [1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]
