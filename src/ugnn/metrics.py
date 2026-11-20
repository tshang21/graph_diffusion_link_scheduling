"""Score (T, K) schedules on their conflict graphs after resolving collisions.

Scoring is STRICT: a link counts only if no active neighbour conflicts, so a collision kills both
endpoints.  Independent-set schedules (MWIS / MIS) score the same under any collision rule.
"""
import numpy as np

from common.utils import effective_success_trajectory
from ugnn.dataset import graph_A


def aggregate_method(schedules, graphs, delta):
    """Mean curves over graphs plus the pooled per-link rates.

    obj_per_t   % of links transmitting successfully per slot (obj_all: the per-graph curves)
    cov_per_t   % of links served at least once by slot t
    viol_per_t  % of links below the running quota delta*(t+1) at slot t
    rates_all   every link's full-horizon success rate, pooled over graphs
    """
    obj, cov, viol, rates = [], [], [], []
    for s, g in zip(schedules, graphs):
        A, s = graph_A(g).numpy(), np.asarray(s, dtype=np.float32)
        cf = effective_success_trajectory(s, A)                # (T, K) collision-free, strict
        T, K = cf.shape
        cum = cf.cumsum(0)
        obj.append(cf.sum(1) * 100.0 / K)
        cov.append((cum > 0).mean(1) * 100.0)
        viol.append((cum < (delta * np.arange(1, T + 1))[:, None]).mean(1) * 100.0)
        rates.append(cum[-1] / T)
    obj = np.stack(obj)
    return {"obj_per_t": obj.mean(0), "obj_all": obj, "cov_per_t": np.mean(cov, 0),
            "viol_per_t": np.mean(viol, 0), "rates_all": np.concatenate(rates)}
