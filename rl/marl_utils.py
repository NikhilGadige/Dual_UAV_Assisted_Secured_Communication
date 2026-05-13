import numpy as np

from core.observation_schema import (
    GEOM_END, DIST_END, CHAN_END, BATT_END, EH_END, NTN_END,
    OFF_H_UR, OFF_H_RB, OFF_H_UE, OFF_H_JE,
    OFF_GAMMA_UR, OFF_GAMMA_RB, OFF_GAMMA_E,
    OFF_R_LEGIT, OFF_R_EVE, OFF_R_SEC, OFF_JAMMER_PWR,
    RELAY_BATT_IDX, JAMMER_BATT_IDX, EH_START, NTN_START,
)


def relay_split_indices(n: int) -> np.ndarray:
    geom = list(range(0, 21))
    dist = list(range(21, 25))
    legit_feats = [25 + OFF_H_UR, 25 + OFF_H_RB,
                   25 + OFF_GAMMA_UR, 25 + OFF_GAMMA_RB,
                   25 + OFF_R_LEGIT, 25 + OFF_R_SEC]
    batt = [RELAY_BATT_IDX]
    idx = geom + dist + legit_feats + batt
    if n > BATT_END:
        idx += list(range(EH_START, min(n, NTN_START)))
    if n > EH_END:
        idx += list(range(NTN_START, n))
    return np.array(idx, dtype=np.int32)


def jammer_split_indices(n: int) -> np.ndarray:
    geom = list(range(0, 21))
    dist = list(range(21, 25))
    eve_feats = [25 + OFF_H_UE, 25 + OFF_H_JE,
                 25 + OFF_GAMMA_E, 25 + OFF_R_EVE,
                 25 + OFF_JAMMER_PWR, 25 + OFF_R_SEC]
    batt = [JAMMER_BATT_IDX]
    idx = geom + dist + eve_feats + batt
    if n > BATT_END:
        idx += list(range(EH_START, min(n, NTN_START)))
    if n > EH_END:
        idx += [NTN_START]
    return np.array(idx, dtype=np.int32)


def relay_observation(obs: np.ndarray) -> np.ndarray:
    n = obs.shape[0]
    return obs[relay_split_indices(n)]


def jammer_observation(obs: np.ndarray) -> np.ndarray:
    n = obs.shape[0]
    return obs[jammer_split_indices(n)]


def relay_obs_dim(obs_dim: int) -> int:
    return len(relay_split_indices(obs_dim))


def jammer_obs_dim(obs_dim: int) -> int:
    return len(jammer_split_indices(obs_dim))

_DIRS = [
    np.array([1.0, 0.0], dtype=np.float32),
    np.array([-1.0, 0.0], dtype=np.float32),
    np.array([0.0, 1.0], dtype=np.float32),
    np.array([0.0, -1.0], dtype=np.float32),
    np.array([1.0, 1.0], dtype=np.float32),
    np.array([1.0, -1.0], dtype=np.float32),
    np.array([-1.0, 1.0], dtype=np.float32),
    np.array([-1.0, -1.0], dtype=np.float32),
]
_SPEEDS = [0.0, 0.5, 1.0]
_POWERS = [-1.0, 0.0, 1.0]


def make_relay_action_table() -> list[np.ndarray]:
    table = []
    for speed in _SPEEDS:
        if speed == 0.0:
            table.append(np.zeros(2, dtype=np.float32))
        else:
            for d in _DIRS:
                table.append((speed * d / np.linalg.norm(d)).astype(np.float32))
    return table


def make_jammer_action_table() -> list[np.ndarray]:
    table = []
    for speed in _SPEEDS:
        if speed == 0.0:
            vel = np.zeros(2, dtype=np.float32)
            for p in _POWERS:
                table.append(np.array([vel[0], vel[1], p], dtype=np.float32))
        else:
            for d in _DIRS:
                vel = (speed * d / np.linalg.norm(d)).astype(np.float32)
                for p in _POWERS:
                    table.append(np.array([vel[0], vel[1], p], dtype=np.float32))
    return table


def decode_jammer_action(action_vec: np.ndarray):
    return action_vec[:2], float(action_vec[2])
