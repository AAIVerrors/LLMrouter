"""
Capacity- and Backlog-Aware Quota FLAIR reward.

Implements the interval reward of the note "Capacity- and Backlog-Aware
Quota FLAIR":

  1. Fair load = balanced normalized post-dispatch work
         z_m = (D_m + n_m) / S_m,     S_m = max(mu_m * dt, eps_mu).
     The integer fair quota minimizes the capacity-weighted dispersion
         Phi(k) = sum_m S_m (z_m - zbar)^2,
     equivalently the separable potential
         Psi(k) = sum_m (D_m + k_m)^2 / S_m,
     solved exactly by discrete water filling on the marginal costs
         Delta_m(j) = (2 (D_m + j) + 1) / S_m.
  2. Among equally fair integer quotas (threshold ties), the closest
     representative to the realized counts in l1 distance is used, so
     label-permutation artifacts are never penalized.
  3. Per-used-server mean rewards are interpolated toward the reward
     floor by a single-parameter ReLU penalty (f = FAIR in [0,1]):
         e_m    = min{1, |N_m - k_m| / max{k_m, 1}}
         p_m    = max(0, e_m + f - 1)
         rhat_m = r_floor + (1 - p_m) (rbar_m - r_floor).
  4. Unused servers with positive quota contribute the fractional floor
     weight W = f * H, and the interval reward is the weighted tilted
     (soft-min, beta < 0) aggregate over used scores and floor terms.

All functions are numpy/python only; the caller converts to torch.
Run `python quota_flair.py` to execute the worked-example self-tests.
"""

import heapq
import math

import numpy as np


def fair_quota(D, S, n_total, realized_counts, rel_tol=1e-9):
    """Discrete water-filling fair quota, closest representative.

    Args:
        D: (M,) backlog at the interval boundary (work units, >= 0).
        S: (M,) service budgets mu_hat * dt (> 0).
        n_total: number of arrivals to allocate (int, >= 0).
        realized_counts: (M,) realized assignment counts N_t^m; used only
            to resolve threshold ties toward the closest fair quota.
        rel_tol: relative tolerance for identifying marginal-cost ties.

    Returns:
        (M,) int64 array k with sum(k) == n_total, a member of the
        optimal quota set closest to realized_counts in l1 distance.
    """
    D = np.asarray(D, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    N = np.asarray(realized_counts, dtype=np.int64)
    M = S.shape[0]
    if not (D.shape[0] == M == N.shape[0]):
        raise ValueError("D, S, realized_counts must share length M")
    if np.any(S <= 0.0):
        raise ValueError("service budgets must be positive")
    n_total = int(n_total)
    if n_total <= 0:
        return np.zeros(M, dtype=np.int64)

    def marginal(m, j):
        # Cost increase of giving server m its (j+1)-th quota unit.
        return (2.0 * (D[m] + j) + 1.0) / S[m]

    # Greedy selection of the n_total smallest marginal costs.
    sel = np.zeros(M, dtype=np.int64)
    heap = [(marginal(m, 0), m) for m in range(M)]
    heapq.heapify(heap)
    threshold = 0.0
    for _ in range(n_total):
        cost, m = heapq.heappop(heap)
        sel[m] += 1
        threshold = cost
        heapq.heappush(heap, (marginal(m, sel[m]), m))

    tol = rel_tol * max(1.0, abs(threshold))

    # Forced units are the marginals strictly below the threshold.
    # Within one server the marginal sequence is strictly increasing,
    # so only its last selected unit can sit on the threshold.
    base = sel.copy()
    for m in range(M):
        if base[m] > 0 and abs(marginal(m, base[m] - 1) - threshold) <= tol:
            base[m] -= 1

    # Threshold candidates: every server whose next marginal equals the
    # threshold, whether or not the greedy pass happened to pick it.
    cand = [m for m in range(M) if abs(marginal(m, base[m]) - threshold) <= tol]
    R = n_total - int(base.sum())
    if R > 0:
        # Closest-representative tie-break: adding one unit to server m
        # changes the l1 distance to the realized counts by Gamma_m.
        gam = sorted(
            (abs(int(N[m]) - (int(base[m]) + 1)) - abs(int(N[m]) - int(base[m])), m)
            for m in cand
        )
        for _, m in gam[:R]:
            base[m] += 1

    assert int(base.sum()) == n_total
    return base


def quota_penalties(counts, k, f):
    """Clipped relative quota error e and ReLU penalty p, elementwise."""
    counts = np.asarray(counts, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    e = np.minimum(1.0, np.abs(counts - k) / np.maximum(k, 1.0))
    p = np.maximum(0.0, e + float(f) - 1.0)
    return e, p


def quota_flair_reward(server_means, counts, k, f, beta, r_floor):
    """Weighted tilted (soft-min) interval reward.

    Args:
        server_means: dict {m: mean reward of used server m}.
        counts: (M,) realized counts N_t^m.
        k: (M,) fair quota from fair_quota().
        f: quota tolerance/strength FAIR in [0, 1].
        beta: tilt parameter, beta < 0 (beta ~ 0 falls back to the
            weighted arithmetic mean).
        r_floor: reward floor r_bot with r_floor <= every realized reward.

    Returns:
        (reward, info) where info holds e, p, rhat, H, W.
    """
    counts = np.asarray(counts, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    used = sorted(server_means.keys())
    if not used:
        raise ValueError("quota_flair_reward called with no used server")
    e, p = quota_penalties(counts, k, f)

    rhat = {
        m: float(r_floor) + (1.0 - float(p[m])) * (float(server_means[m]) - float(r_floor))
        for m in used
    }

    miss = [m for m in range(len(k)) if k[m] > 0 and counts[m] == 0]
    H = len(miss)
    W = float(f) * H
    denom = len(used) + W

    if abs(beta) < 1e-8:
        reward = (sum(rhat.values()) + W * float(r_floor)) / denom
    else:
        terms = [beta * rhat[m] for m in used]
        if W > 0.0:
            terms.append(math.log(W) + beta * float(r_floor))
        tmax = max(terms)
        log_z = tmax + math.log(sum(math.exp(t - tmax) for t in terms))
        reward = (log_z - math.log(denom)) / beta

    return float(reward), {"e": e, "p": p, "rhat": rhat, "H": H, "W": W}


def load_fairness(counts, k):
    """Relocation-distance metric F_load = 1 - ||N - k||_1 / (2 N_t).

    Reward-independent: 1 exactly when the realized counts are a fair
    quota; 0 when every request must be relocated.
    """
    counts = np.asarray(counts, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    n = int(counts.sum())
    if n <= 0:
        return 1.0
    return 1.0 - float(np.abs(counts - k).sum()) / (2.0 * n)


def jain_normalized_load(D, S, counts):
    """Jain's index over normalized post-dispatch loads z = (D + n)/S."""
    D = np.asarray(D, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    z = (D + np.asarray(counts, dtype=np.float64)) / S
    denom = len(z) * float(np.square(z).sum())
    if denom <= 0.0:
        return 1.0
    return float(np.square(z.sum()) / denom)


# =====================================================================
# Self-tests: the worked examples of the algorithm note.
# =====================================================================
def _self_test():
    # ---- Example 1: equal servers, indivisible arrivals ------------
    # M=4, Nt=5, D=0, equal budgets; realized (3,1,1,0) -> k=(2,1,1,1).
    D = np.zeros(4)
    S = np.ones(4)
    counts = np.array([3, 1, 1, 0])
    k = fair_quota(D, S, 5, counts)
    assert k.tolist() == [2, 1, 1, 1], k

    means = {0: 0.4, 1: -0.2, 2: 0.1}
    r_half, info_half = quota_flair_reward(means, counts, k, 0.5, -2.0, -0.7)
    r_one, info_one = quota_flair_reward(means, counts, k, 1.0, -2.0, -0.7)
    assert info_half["H"] == 1 and abs(info_half["W"] - 0.5) < 1e-12
    assert abs(r_half - (-0.156)) < 2e-3, r_half   # note: -0.156
    assert abs(r_one - (-0.329)) < 2e-3, r_one     # note: -0.329

    # ---- Example 2: heterogeneous servers with initial backlog ----
    # S=(4,2,2,1), D=(2,0,0,0), Nt=7 -> k=(2,2,2,1), post-loads all 1.
    D2 = np.array([2.0, 0.0, 0.0, 0.0])
    S2 = np.array([4.0, 2.0, 2.0, 1.0])
    counts2 = np.array([2, 2, 2, 1])
    k2 = fair_quota(D2, S2, 7, counts2)
    assert k2.tolist() == [2, 2, 2, 1], k2
    assert abs(load_fairness(counts2, k2) - 1.0) < 1e-12
    assert abs(jain_normalized_load(D2, S2, counts2) - 1.0) < 1e-12

    # ---- Sparse regime Nt < M --------------------------------------
    # Spread (1,1,0,0) is a fair quota: no penalty, no floor.
    Ds = np.zeros(4)
    Ss = np.ones(4)
    spread = np.array([1, 1, 0, 0])
    ks = fair_quota(Ds, Ss, 2, spread)
    assert ks.tolist() == [1, 1, 0, 0], ks
    r_spread, info_s = quota_flair_reward({0: 0.2, 1: 0.2}, spread, ks, 0.5, -2.0, -0.7)
    assert info_s["H"] == 0 and abs(r_spread - 0.2) < 1e-9

    # Sharing (2,0,0,0): quota penalty on server 0 plus one floor term.
    shared = np.array([2, 0, 0, 0])
    kc = fair_quota(Ds, Ss, 2, shared)
    assert kc.sum() == 2 and kc.max() == 1, kc
    r_shared, info_c = quota_flair_reward({0: 0.2}, shared, kc, 0.5, -2.0, -0.7)
    assert info_c["H"] == 1
    assert r_shared < r_spread, (r_shared, r_spread)

    # ---- Dense regime: the (91,1,...,1) reviewer counterexample ----
    M = 10
    Dd = np.zeros(M)
    Sd = np.ones(M)
    balanced = np.full(M, 10)
    conc = np.array([91] + [1] * 9)
    kb = fair_quota(Dd, Sd, 100, balanced)
    kcnc = fair_quota(Dd, Sd, 100, conc)
    assert kb.tolist() == [10] * M and kcnc.tolist() == [10] * M
    same_means = {m: 0.2 for m in range(M)}
    r_bal, _ = quota_flair_reward(same_means, balanced, kb, 0.5, -2.0, -0.7)
    r_conc, _ = quota_flair_reward(same_means, conc, kcnc, 0.5, -2.0, -0.7)
    assert abs(r_bal - 0.2) < 1e-9
    assert r_conc < r_bal - 0.1, (r_conc, r_bal)
    assert load_fairness(conc, kcnc) < load_fairness(balanced, kb)

    print("quota_flair self-tests passed:")
    print(f"  example 1: f=0.5 -> {r_half:.4f} (note -0.156), "
          f"f=1 -> {r_one:.4f} (note -0.329)")
    print(f"  sparse: spread {r_spread:.4f} vs shared {r_shared:.4f}")
    print(f"  dense: balanced {r_bal:.4f} vs (91,1,...,1) {r_conc:.4f}, "
          f"F_load {load_fairness(conc, kcnc):.3f}")


if __name__ == "__main__":
    _self_test()