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

from config import Config


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
        if str(getattr(Config, "QUOTA_TIEBREAK", "lex")).lower() == "lex":
            # Canonical representative among the optimal quota set:
            #   1) minimize Psi (all candidates already do),
            #   2) among them MAXIMIZE the reference variance Dbar,
            #   3) lexicographic endpoint id only for exact remaining ties.
            # Step 2 makes the choice invariant to endpoint relabeling (Dbar
            # is permutation-invariant) and avoids picking a degenerate
            # representative: with S=(1,3), N=2 the two optima carry
            # Dbar = 0.667 and 0.0, and pure lex could land on 0. The choice
            # must also not depend on the realized counts, or the reference
            # scale itself becomes action-dependent.
            # Adding one unit to server m changes its Dbar contribution by
            #   [1 - (2*base_m + 1)/N] / S_m.
            gains = sorted(
                ((-(1.0 - (2.0 * base[m] + 1.0) / max(n_total, 1)) / S[m], m)
                 for m in cand)
            )
            for _, m in gains[:R]:
                base[m] += 1
        else:
            # Legacy closest-representative tie-break: adding one unit to
            # server m changes the l1 distance to the realized counts by
            # Gamma_m. Kept for reproducing old runs only; it makes Dbar
            # depend on the actions being scored.
            gam = sorted(
                (abs(int(N[m]) - (int(base[m]) + 1)) - abs(int(N[m]) - int(base[m])), m)
                for m in cand
            )
            for _, m in gam[:R]:
                base[m] += 1

    assert int(base.sum()) == n_total
    return base


# =====================================================================
# Water-filling regret fairness (constrained / RCPO formulation)
# =====================================================================

def wf_psi(D, S, n):
    """Water-filling cost Psi(n) = sum_m (D_m + n_m)^2 / S_m."""
    D = np.asarray(D, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    return float(np.sum((D + n) ** 2 / S))


def wf_regret_stats(D, S, counts, k):
    """Observed water-filling regret and its i.i.d. reference normalization.

    Returns (delta_obs, dbar, v_obs, g):
      delta_obs = Psi(counts) - Psi(k)                       >= 0
      dbar      = sum_m k_m (1 - k_m/N) / S_m
                  = E[delta_obs] under counts ~ Multinomial(N, k/N),
                  the i.i.d. QUOTA-PROPORTIONAL SAMPLING REFERENCE -- not a
                  floor: the true minimum is 0 (counts == k), and a
                  deterministic conditional policy whose marginals hit the
                  quota beats the reference (v -> 0).
      v_obs     = delta_obs / dbar   (np.nan when dbar degenerates)
      g         = delta_obs / (Psi_max - Psi(k)), in [0, 1]: the fraction of
                  the worst-case water-filling loss actually incurred, where
                  Psi_max routes every request to the single worst endpoint.
                  Reported alongside v because it needs no reference scale.
    """
    D = np.asarray(D, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    N = float(counts.sum())

    psi_k = wf_psi(D, S, k)
    delta_obs = max(wf_psi(D, S, counts) - psi_k, 0.0)
    dbar = float(np.sum(k * (1.0 - k / max(N, 1.0)) / S))

    base = float(np.sum(D ** 2 / S))
    psi_max = base + float(np.max(((D + N) ** 2 - D ** 2) / S))
    denom_g = max(psi_max - psi_k, 1e-12)
    g = float(min(max(delta_obs / denom_g, 0.0), 1.0))

    v_obs = float(delta_obs / dbar) if dbar > 0.0 else float("nan")
    return delta_obs, dbar, v_obs, g


def wf_expected_regret(D, S, probs, k, dbar):
    """Rao-Blackwellized expected regret of the CONDITIONAL policy.

    probs: (N, M) per-request routing probabilities pi_i. With requests
    conditionally independent given (s_t, p_{1:N}) [A2]:

        E_pi[Delta^Psi | s, p] = sum_m [ (D+nhat_m)^2 + sig2_m - (D+k_m)^2 ] / S_m
        nhat_m = sum_i pi_{i,m},   sig2_m = sum_i pi_{i,m}(1 - pi_{i,m})

    (verified against Monte Carlo to <0.4%). Used for the DUAL update: it is
    the exact conditional expectation, so the multiplier tracks the policy's
    expected constraint cost instead of one noisy multinomial draw. The actor
    reward keeps the sampled v_obs, which preserves cross-interval credit
    (routing now -> backlog next interval -> future regret).

    Returns (v_hat, v_mean_allocation, v_sampling), each normalized by dbar:
      v_mean_allocation = [Psi(nhat) - Psi(k)] / dbar   expected-allocation term;
                          MAY be negative (nhat is a continuous vector and can
                          undercut the integer optimum)
      v_sampling        = [sum_m sig2_m / S_m] / dbar   policy randomness

    v_hat itself is NONNEGATIVE by construction: it equals
    E[Psi(X) - Psi(k)] and every realizable integer count vector X is a
    feasible solution of the water-filling problem, so Psi(X) >= Psi(k)
    pointwise. The sampling term exactly repays whatever the continuous
    relaxation undercuts. A materially negative v_hat is therefore an
    implementation error, not numerics -- it is raised, not clipped.
    """
    D = np.asarray(D, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    P = np.asarray(probs, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    if dbar <= 0.0 or P.size == 0:
        return float("nan"), float("nan"), float("nan")
    nhat = P.sum(axis=0)
    sig2 = (P * (1.0 - P)).sum(axis=0)
    mean_alloc = wf_psi(D, S, nhat) - wf_psi(D, S, k)
    sampling = float(np.sum(sig2 / S))
    v_hat = float((mean_alloc + sampling) / dbar)
    if v_hat < -1e-7:
        raise AssertionError(
            f"wf_expected_regret: v_hat={v_hat} < 0 is impossible "
            "(pointwise Psi(X) >= Psi(k)); check D/S/k/probs consistency"
        )
    return (
        max(v_hat, 0.0),
        float(mean_alloc / dbar),
        float(sampling / dbar),
    )


def quota_penalties(counts, k, f):
    """Clipped relative quota error e and ReLU penalty p, elementwise."""
    counts = np.asarray(counts, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    e = np.minimum(1.0, np.abs(counts - k) / np.maximum(k, 1.0))
    p = np.maximum(0.0, e + float(f) - 1.0)
    return e, p


def quota_flair_reward(server_means, counts, k, f, beta, r_floor, mu=1.0):
    """Weighted tilted (soft-min) interval reward.

    Args:
        server_means: dict {m: mean reward of used server m}.
        counts: (M,) realized counts N_t^m.
        k: (M,) fair quota from fair_quota().
        f: quota tolerance/strength FAIR in [0, 1].
        beta: tilt parameter, beta < 0 (beta ~ 0 falls back to the
            weighted arithmetic mean).
        r_floor: reward floor r_bot with r_floor <= every realized reward.
        mu: Lagrangian multiplier scaling the quota penalty (>= 0). mu=1
            reproduces the fixed-strength reward. Unlike f, mu is not capped
            at 1: mu*p > 1 drives an over-allocated server BELOW r_floor,
            which is the extra pressure a constrained formulation needs to
            hold a fairness floor that fixed f cannot reach.

    Returns:
        (reward, info) where info holds e, p, rhat, H, W.
    """
    counts = np.asarray(counts, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    used = sorted(server_means.keys())
    if not used:
        raise ValueError("quota_flair_reward called with no used server")
    e, p = quota_penalties(counts, k, f)
    p = float(max(0.0, mu)) * p

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