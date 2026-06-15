#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parallel_kl_probe.py
====================
诊断实验:验证 "联合-边缘 KL(over-denoising)" 是否携带 confidence 看不见的信号。

核心论点(对应你和 Apple-2512.09106 / dUltra-2512.21446 的缝):
  现有揭示决策都建立在【边缘量】上(confidence / 独立 Bernoulli / 散布度代理)。
  对一对 minimal pair:
    - "bound"   槽(如 am/a):两个槽边缘各自高 confidence,但联合被语法绑死
    - "unbound" 槽(如 am/tired):同样的边缘 confidence,但两槽无关
  confidence 对这两类【相同】,但"该不该并行揭示"的答案【相反】。
  只有 KL( joint || marginal_product ) 能把它们分开。

本文件做两件事:
  1) --selftest : 在【已知答案】的合成 joint 上验证度量正确
                  (对 coherent joint, KL(joint||product) == 互信息 I(A;B),可解析核对)。
                  这一步在任何机器上都能跑,用来证明"度量本身没写错"。
  2) --model    : 把同一套 core 套到真实 MDLM(LLaDA 风格)上,在你的 p2l 环境跑。

度量定义(两个 mask 槽 A、B,候选集已限制到 top-k 便于精确枚举):
  marg_A = p(A | both masked)              # 单次 forward 的边缘
  marg_B = p(B | both masked)
  cond_B_given_A[a] = p(B | A=a, B masked) # 填回 A 再 forward
  joint_seq(a,b)    = marg_A[a] * cond_B_given_A[a][b]   # 模型自洽的"顺序联合"(一种 ordering)
  product(a,b)      = marg_A[a] * marg_B[b]              # 并行揭示实际采样的分布
  over_denoising_KL = KL( joint_seq || product )         # 并行引入的分布偏差
  incoherence_mass  = 并行采样落在 joint_seq≈0 的 pair 上的概率(可解释的"错误率")
  confidence        = max(marg)  per slot                # baseline 用的信号

注意:MDLM 的 joint 是 order-dependent(A→B 与 B→A 可能不同 —— 对应
2602.00286 的 path-dependent divergence),所以两个方向都报,并报平均。
"""

import argparse
import numpy as np

EPS = 1e-12


# ============================================================================
# Section 1 — model-agnostic core (纯数组运算,可单测)
# ============================================================================
def _norm(p):
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    return p / s if s > 0 else p


def kl(p, q):
    """KL(p || q) in nats, over the shared grid. 0*log0=0; guard q=0 where p>0."""
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    mask = p > EPS
    if np.any((q <= EPS) & mask):
        # parallel puts ~0 mass where the joint has mass -> formally infinite;
        # clamp q so the number stays finite but large (and flag via incoherence_mass).
        q = np.clip(q, EPS, None)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def mutual_information(joint):
    """I(A;B) in nats for a normalized 2D joint. Equals KL(joint || marg_outer)."""
    J = _norm(np.asarray(joint, np.float64))
    mA = J.sum(axis=1, keepdims=True)
    mB = J.sum(axis=0, keepdims=True)
    prod = mA @ mB
    return kl(J.ravel(), prod.ravel())


def analyze_slot_pair(marg_A, cond_B_given_A, marg_B, cond_A_given_B,
                      labels_A=None, labels_B=None, eps_incoh=0.02):
    """
    All inputs are over the (already chosen) candidate grids.
      marg_A: [kA]            marg_B: [kB]
      cond_B_given_A: [kA,kB] (row a = p(B|A=a))
      cond_A_given_B: [kB,kA] (row b = p(A|B=b))
    Returns a dict of the signals.
    """
    marg_A = _norm(marg_A); marg_B = _norm(marg_B)
    cBA = np.vstack([_norm(r) for r in cond_B_given_A])      # [kA,kB]
    cAB = np.vstack([_norm(r) for r in cond_A_given_B])      # [kB,kA]

    # two sequential joints (two orderings)
    joint_AB = marg_A[:, None] * cBA                          # [kA,kB]
    joint_BA = (marg_B[:, None] * cAB).T                      # -> [kA,kB]
    product  = marg_A[:, None] * marg_B[None, :]              # [kA,kB]

    kl_AB = kl(joint_AB.ravel(), product.ravel())
    kl_BA = kl(joint_BA.ravel(), product.ravel())
    kl_avg = 0.5 * (kl_AB + kl_BA)

    # interpretable error rate: parallel mass on pairs the joint deems ~impossible
    Jref = 0.5 * (joint_AB + joint_BA)
    dead = Jref < (eps_incoh * Jref.max())
    incoh_mass = float(product[dead].sum())

    return {
        "conf_A": float(marg_A.max()),
        "conf_B": float(marg_B.max()),
        "min_conf": float(min(marg_A.max(), marg_B.max())),
        "kl_AB_nats": kl_AB,
        "kl_BA_nats": kl_BA,
        "kl_avg_nats": kl_avg,
        "kl_avg_bits": kl_avg / np.log(2),
        "incoherence_mass": incoh_mass,
    }


def auroc(scores, labels):
    """Tiny AUROC: P(score(pos) > score(neg)). labels in {0,1}. Ties=0.5."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for a in pos:
        wins += np.sum(a > neg) + 0.5 * np.sum(a == neg)
    return wins / (len(pos) * len(neg))


# ============================================================================
# Section 2 — synthetic self-test (已知答案,证明 core 正确)
# ============================================================================
def _coherent_from_joint(J):
    """Given a true 2D joint, derive the marginals & conditionals a *coherent*
    model would return. For such inputs both orderings agree and
    KL(joint||product) == I(A;B) exactly."""
    J = _norm(J)
    mA = J.sum(axis=1)                       # [kA]
    mB = J.sum(axis=0)                        # [kB]
    cBA = J / np.clip(mA[:, None], EPS, None) # p(B|A)
    cAB = (J / np.clip(mB[None, :], EPS, None)).T  # p(A|B), [kB,kA]
    return mA, cBA, mB, cAB


def selftest():
    print("=" * 74)
    print("SELF-TEST  (synthetic joints with known answers)")
    print("=" * 74)

    # ---- bound: marginals (0.9, 0.85), maximal positive coupling (am/a) ----
    # Fréchet-upper joint with these marginals:
    #            b=a     b=an
    #  a=am     0.85    0.05      (row 0.90)
    #  a=are    0.00    0.10      (row 0.10)
    #  col      0.85    0.15
    J_bound = np.array([[0.85, 0.05],
                        [0.00, 0.10]])
    # ---- unbound: SAME marginals but independent (am/tired) ----
    mA = J_bound.sum(1); mB = J_bound.sum(0)
    J_unbound = np.outer(mA, mB)

    rows = []
    for name, J, lab in [("am_a   (bound)", J_bound, 1),
                         ("am_tired (unbound)", J_unbound, 0)]:
        a, cBA, b, cAB = _coherent_from_joint(J)
        r = analyze_slot_pair(a, cBA, b, cAB)
        r["name"] = name; r["label"] = lab
        rows.append(r)

        mi = mutual_information(J)
        # checks: confidence identical across the two; KL == MI for coherent joint
        assert abs(r["conf_A"] - 0.9) < 1e-9 and abs(r["conf_B"] - 0.85) < 1e-9
        assert abs(r["kl_AB_nats"] - r["kl_BA_nats"]) < 1e-9, "orderings must agree for coherent joint"
        assert abs(r["kl_avg_nats"] - mi) < 1e-9, f"KL({r['kl_avg_nats']:.6f}) != MI({mi:.6f})"

    _print_table(rows)
    _print_separation(rows)

    kl_bound = rows[0]["kl_avg_nats"]; kl_unbound = rows[1]["kl_avg_nats"]
    print("\nchecks passed:")
    print(f"  - both examples share identical confidence (0.90, 0.85)")
    print(f"  - KL(bound)={kl_bound:.4f} nats  vs  KL(unbound)={kl_unbound:.4f} nats")
    print(f"  - KL == analytic mutual information (coherent-joint identity)  [OK]")
    print(f"  - min_conf is IDENTICAL across the pair -> confidence cannot separate")
    print(f"  - KL separates them cleanly            -> the signal is real")
    print(f"  - parallel incoherence mass(bound) = {rows[0]['incoherence_mass']*100:.1f}%"
          f"  (am/an + are/a that the joint forbids)")


# ============================================================================
# Section 3 — real MDLM harness (在 p2l 环境跑;此沙盒无法下载权重)
# ============================================================================
def run_real_model(model_path, mask_token_id, device="cuda"):
    """
    Wire to a LLaDA-style MDLM. Each probe: a template with exactly two slots,
    each slot's candidates must tokenize to a SINGLE token (so the joint is
    enumerable & exact). Adapt `PROBES` to your tokenizer's spacing rules.
    """
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    except Exception:
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    if mask_token_id is None:
        mask_token_id = getattr(tok, "mask_token_id", None)
        if mask_token_id is None:
            raise SystemExit("Set --mask-token-id (LLaDA-8B-Instruct commonly uses 126336).")

    def single_tok(word):
        ids = tok.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{word!r} -> {len(ids)} tokens (need exactly 1)")
        return ids[0]

    @torch.no_grad()
    def dist_at(ids_list, pos, cand_ids):
        ids = torch.tensor([ids_list], device=device)
        logits = model(ids).logits[0, pos]           # [V]
        probs = torch.softmax(logits.float(), -1)
        return probs[cand_ids].cpu().numpy()

    # ---- EDIT THESE PROBES ----------------------------------------------
    # template: list of token-strings; use "<A>" / "<B>" to mark the two slots.
    PROBES = [
        dict(name="am/a (bound)",   label=1,
             template=["i", "<A>", "<B>", "student"],
             cand_A=[" am", " are"], cand_B=[" a", " an"]),
        dict(name="am/tired (unbound)", label=0,
             template=["i", "<A>", "very", "<B>"],
             cand_A=[" am", " are"], cand_B=[" tired", " happy"]),
        # add more matched pairs; aim to match confidences across bound/unbound.
    ]
    # ---------------------------------------------------------------------

    rows = []
    for p in PROBES:
        candA = [single_tok(w) for w in p["cand_A"]]
        candB = [single_tok(w) for w in p["cand_B"]]
        base, posA, posB = [], None, None
        for t in p["template"]:
            if t == "<A>":
                posA = len(base); base.append(mask_token_id)
            elif t == "<B>":
                posB = len(base); base.append(mask_token_id)
            else:
                ids = tok.encode(t if not base else " " + t, add_special_tokens=False)
                base.extend(ids)
        # both-masked marginals
        mA = dist_at(base, posA, candA)
        mB = dist_at(base, posB, candB)
        # conditionals: fill one slot with each candidate, re-forward
        cBA = []
        for a in candA:
            s = list(base); s[posA] = a
            cBA.append(dist_at(s, posB, candB))
        cAB = []
        for b in candB:
            s = list(base); s[posB] = b
            cAB.append(dist_at(s, posA, candA))
        r = analyze_slot_pair(mA, np.array(cBA), mB, np.array(cAB))
        r["name"] = p["name"]; r["label"] = p["label"]
        rows.append(r)

    _print_table(rows)
    _print_separation(rows)


# ============================================================================
# Section 4 — reporting
# ============================================================================
def _print_table(rows):
    print(f"\n{'probe':<22}{'conf_A':>8}{'conf_B':>8}{'min_conf':>10}"
          f"{'KL(nats)':>10}{'KL(bits)':>10}{'incoh%':>9}")
    print("-" * 77)
    for r in rows:
        print(f"{r['name']:<22}{r['conf_A']:>8.3f}{r['conf_B']:>8.3f}"
              f"{r['min_conf']:>10.3f}{r['kl_avg_nats']:>10.4f}"
              f"{r['kl_avg_bits']:>10.4f}{r['incoherence_mass']*100:>8.1f}%")


def _print_separation(rows):
    labels = [r["label"] for r in rows]
    if len(set(labels)) < 2:
        return
    kl_auc = auroc([r["kl_avg_nats"] for r in rows], labels)
    cf_auc = auroc([r["min_conf"] for r in rows], labels)
    print("\nseparation of bound(1) vs unbound(0)  [AUROC, 1.0=perfect, 0.5=chance]")
    print(f"  KL signal      : {kl_auc:.3f}")
    print(f"  min_conf signal: {cf_auc:.3f}")
    print("  -> if KL>>conf, KL carries信号 the marginals don't.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run synthetic validation (no model needed)")
    ap.add_argument("--model", default=None, help="HF path/id of a LLaDA-style MDLM")
    ap.add_argument("--mask-token-id", type=int, default=None,
                    help="mask id (LLaDA-8B-Instruct: 126336)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.selftest or not args.model:
        selftest()
    if args.model:
        run_real_model(args.model, args.mask_token_id, args.device)


if __name__ == "__main__":
    main()