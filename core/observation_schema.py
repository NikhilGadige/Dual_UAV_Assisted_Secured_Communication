# Slice endpoints (base, without multi-Eve agg features)
GEOM_END = 21       # 0:21
DIST_END = 25       # 21:25
CHAN_END = 36       # 25:36
BATT_END = 38       # 36:38
EH_END = 43         # 38:43  (only in full_eh / full_ntn mode)
NTN_END = 46        # 43:46  (only in full_ntn mode)
# Multi-Eve aggregated features appended after base: 4 values
EVE_AGG_N = 4

# Channel sub-indices (offsets within [25:36))
OFF_H_UR = 0
OFF_H_RB = 1
OFF_H_UE = 2
OFF_H_JE = 3
OFF_GAMMA_UR = 4
OFF_GAMMA_RB = 5
OFF_GAMMA_E = 6
OFF_R_LEGIT = 7
OFF_R_EVE = 8
OFF_R_SEC = 9
OFF_JAMMER_PWR = 10

# Battery absolute indices
RELAY_BATT_IDX = 36
JAMMER_BATT_IDX = 37

# EH absolute indices (full_eh / full_ntn only)
EH_START = 38

# NTN absolute indices (full_ntn only)
NTN_START = 43
# NTN features: [sat_elevation_rad, sat_relay_dist_norm, h_sat_relay_norm]

# Multi-Eve aggregated feature offsets (relative to end of base vector)
EVE_AGG_NEAREST_DIST = 0
EVE_AGG_MEAN_DIST = 1
EVE_AGG_MAX_CAPACITY = 2
EVE_AGG_NUM_EVES = 3
