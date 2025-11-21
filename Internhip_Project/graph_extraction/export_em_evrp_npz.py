from pathlib import Path
import numpy as np


def save_em_evrp_npz(out_path: Path,
                      xy: np.ndarray,
                      D: np.ndarray,
                      T: np.ndarray,
                      S: np.ndarray,
                      types: np.ndarray,
                      charging_num: int,
                      demands: np.ndarray,
                      poi_node_ids: np.ndarray = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'xy': xy.astype(np.float32),
        'D': D.astype(np.float32),
        'T': T.astype(np.float32),
        'S': S.astype(np.float32),
        'types': types.astype(np.int32),
        'charging_num': int(charging_num),
        'demands': demands.astype(np.float32)
    }
    if poi_node_ids is not None:
        save_dict['poi_node_ids'] = poi_node_ids.astype(np.int64)
    np.savez_compressed(out_path, **save_dict)
