from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    prompt: str = "a red fox in snow, cinematic, highly detailed"
    negative_prompt: str = ""
    output_dir: str = "runs/parallel_window_verify_sd15"

    height: int = 512
    width: int = 512
    num_inference_steps: int = 16
    num_macro_steps: int = 4
    verify_window_size: int = 4
    guidance_scale: float = 7.5
    seed: int = 123

    device: str = "cuda"
    dtype: str = "float16"

    tau: float = 0.05
    save_images: bool = True


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def torch_dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def maybe_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def rel_l2_error(x_hat: torch.Tensor, x_ref: torch.Tensor, eps: float = 1e-8) -> float:
    num = torch.norm((x_hat - x_ref).float()).item()
    den = torch.norm(x_ref.float()).item() + eps
    return float(num / den)


def rel_l2_error_batch(x_hat: torch.Tensor, x_ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.norm((x_hat - x_ref).float().flatten(1), dim=1)
    den = torch.norm(x_ref.float().flatten(1), dim=1) + eps
    return num / den


# ============================================================
# Prompt / Latents / Decode
# ============================================================

@torch.no_grad()
def encode_prompt_pair(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    cond_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    cond_ids = cond_inputs.input_ids.to(device)
    cond = text_encoder(cond_ids)[0].to(device=device, dtype=dtype)

    uncond_inputs = tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_ids = uncond_inputs.input_ids.to(device)
    uncond = text_encoder(uncond_ids)[0].to(device=device, dtype=dtype)

    return uncond, cond


@torch.no_grad()
def prepare_latents(
    pipe: StableDiffusionPipeline,
    cfg: Config,
    generator: torch.Generator,
) -> torch.Tensor:
    vae_scale_factor = pipe.vae_scale_factor
    shape = (
        1,
        pipe.unet.config.in_channels,
        cfg.height // vae_scale_factor,
        cfg.width // vae_scale_factor,
    )
    latents = torch.randn(
        shape,
        generator=generator,
        device=cfg.device,
        dtype=torch_dtype_from_string(cfg.dtype),
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    return latents


@torch.no_grad()
def decode_latents_to_pil(
    pipe: StableDiffusionPipeline,
    latents: torch.Tensor,
) -> Image.Image:
    latents = latents.to(device=pipe.device, dtype=pipe.vae.dtype)
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image


# ============================================================
# DDIM helpers
# ============================================================

def get_alpha_prod_single(
    scheduler: DDIMScheduler,
    timesteps: torch.Tensor,
    pos: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if pos >= len(timesteps):
        alpha = scheduler.final_alpha_cumprod
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        return alpha.to(device=device, dtype=dtype)

    t_val = int(timesteps[pos].item())
    alpha = scheduler.alphas_cumprod[t_val]
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha)
    return alpha.to(device=device, dtype=dtype)


def get_alpha_prod_batch_from_positions(
    scheduler: DDIMScheduler,
    timesteps: torch.Tensor,
    positions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    alphas = []
    n_total = len(timesteps)
    for pos in positions.tolist():
        if pos >= n_total:
            a = scheduler.final_alpha_cumprod
        else:
            t_val = int(timesteps[pos].item())
            a = scheduler.alphas_cumprod[t_val]
        if not torch.is_tensor(a):
            a = torch.tensor(a)
        alphas.append(a.to(device=device, dtype=dtype))
    return torch.stack(alphas, dim=0).view(-1, 1, 1, 1)


@torch.no_grad()
def predict_noise_batch(
    pipe: StableDiffusionPipeline,
    latents: torch.Tensor,
    timesteps_batch: torch.Tensor,
    uncond_embed: torch.Tensor,
    cond_embed: torch.Tensor,
    guidance_scale: float,
    stats: Dict[str, float],
) -> torch.Tensor:
    batch_size = latents.shape[0]

    if guidance_scale > 1.0:
        model_input = torch.cat([latents, latents], dim=0)
        timestep_input = torch.cat([timesteps_batch, timesteps_batch], dim=0)
        encoder_hidden_states = torch.cat(
            [
                uncond_embed.expand(batch_size, -1, -1),
                cond_embed.expand(batch_size, -1, -1),
            ],
            dim=0,
        )
    else:
        model_input = latents
        timestep_input = timesteps_batch
        encoder_hidden_states = cond_embed.expand(batch_size, -1, -1)

    model_input = pipe.scheduler.scale_model_input(model_input, timestep_input)

    noise_pred = pipe.unet(
        model_input,
        timestep_input,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
    )[0]

    stats["unet_calls"] += 1.0

    if guidance_scale > 1.0:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return noise_pred


@torch.no_grad()
def predict_noise_single(
    pipe: StableDiffusionPipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    uncond_embed: torch.Tensor,
    cond_embed: torch.Tensor,
    guidance_scale: float,
    stats: Dict[str, float],
) -> torch.Tensor:
    return predict_noise_batch(
        pipe=pipe,
        latents=latents,
        timesteps_batch=timestep.reshape(1),
        uncond_embed=uncond_embed,
        cond_embed=cond_embed,
        guidance_scale=guidance_scale,
        stats=stats,
    )


@torch.no_grad()
def ddim_direct_jump_single(
    scheduler: DDIMScheduler,
    x_t: torch.Tensor,
    eps_t: torch.Tensor,
    current_timestep_value: int,
    target_alpha_prod: torch.Tensor,
) -> torch.Tensor:
    a_t = scheduler.alphas_cumprod[current_timestep_value]
    if not torch.is_tensor(a_t):
        a_t = torch.tensor(a_t)
    a_t = a_t.to(device=x_t.device, dtype=x_t.dtype).view(1, 1, 1, 1)
    a_s = target_alpha_prod.to(device=x_t.device, dtype=x_t.dtype).view(1, 1, 1, 1)

    sqrt_a_t = torch.sqrt(a_t)
    sqrt_1m_a_t = torch.sqrt(1.0 - a_t)
    sqrt_a_s = torch.sqrt(a_s)
    sqrt_1m_a_s = torch.sqrt(1.0 - a_s)

    x0_hat = (x_t - sqrt_1m_a_t * eps_t) / torch.clamp(sqrt_a_t, min=1e-8)
    x_s = sqrt_a_s * x0_hat + sqrt_1m_a_s * eps_t
    return x_s


@torch.no_grad()
def batched_ddim_step_positions(
    scheduler: DDIMScheduler,
    x_t: torch.Tensor,
    eps_t: torch.Tensor,
    timesteps: torch.Tensor,
    current_positions: torch.Tensor,
    next_positions: torch.Tensor,
) -> torch.Tensor:
    a_t = get_alpha_prod_batch_from_positions(
        scheduler, timesteps, current_positions, x_t.device, x_t.dtype
    )
    a_s = get_alpha_prod_batch_from_positions(
        scheduler, timesteps, next_positions, x_t.device, x_t.dtype
    )

    sqrt_a_t = torch.sqrt(a_t)
    sqrt_1m_a_t = torch.sqrt(1.0 - a_t)
    sqrt_a_s = torch.sqrt(a_s)
    sqrt_1m_a_s = torch.sqrt(1.0 - a_s)

    x0_hat = (x_t - sqrt_1m_a_t * eps_t) / torch.clamp(sqrt_a_t, min=1e-8)
    x_s = sqrt_a_s * x0_hat + sqrt_1m_a_s * eps_t
    return x_s


# ============================================================
# Baseline
# ============================================================

@torch.no_grad()
def run_baseline(
    pipe: StableDiffusionPipeline,
    latents0: torch.Tensor,
    uncond_embed: torch.Tensor,
    cond_embed: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    stats = {"unet_calls": 0.0}
    latents = latents0.clone()

    pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=cfg.device)
    timesteps = pipe.scheduler.timesteps

    maybe_sync(cfg.device)
    t0 = time.perf_counter()

    for pos in range(len(timesteps)):
        eps = predict_noise_single(
            pipe,
            latents,
            timesteps[pos].reshape(1),
            uncond_embed,
            cond_embed,
            cfg.guidance_scale,
            stats,
        )
        out = pipe.scheduler.step(eps, timesteps[pos], latents, return_dict=True)
        latents = out.prev_sample

    maybe_sync(cfg.device)
    stats["wall_time_sec"] = float(time.perf_counter() - t0)
    return latents, stats


# ============================================================
# Windowed draft + parallel local verify + prefix commit
# ============================================================

@torch.no_grad()
def build_window_coarse_trajectory(
    pipe: StableDiffusionPipeline,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    start_macro_idx: int,
    num_blocks: int,
    k_per_block: int,
    uncond_embed: torch.Tensor,
    cond_embed: torch.Tensor,
    cfg: Config,
    stats: Dict[str, float],
) -> List[torch.Tensor]:
    x_hat: List[torch.Tensor] = [x_start.clone()]

    for local_block_idx in range(num_blocks):
        block_start_pos = (start_macro_idx + local_block_idx) * k_per_block
        block_end_pos = block_start_pos + k_per_block

        current_t = timesteps[block_start_pos]
        current_t_val = int(current_t.item())
        target_alpha = get_alpha_prod_single(
            pipe.scheduler,
            timesteps,
            block_end_pos,
            x_start.device,
            x_start.dtype,
        )

        eps = predict_noise_single(
            pipe,
            x_hat[local_block_idx],
            current_t.reshape(1),
            uncond_embed,
            cond_embed,
            cfg.guidance_scale,
            stats,
        )
        x_next = ddim_direct_jump_single(
            pipe.scheduler,
            x_hat[local_block_idx],
            eps,
            current_t_val,
            target_alpha,
        )
        x_hat.append(x_next)

    return x_hat


@torch.no_grad()
def verify_window_parallel_local(
    pipe: StableDiffusionPipeline,
    x_hat: List[torch.Tensor],
    timesteps: torch.Tensor,
    start_macro_idx: int,
    num_blocks: int,
    k_per_block: int,
    uncond_embed: torch.Tensor,
    cond_embed: torch.Tensor,
    cfg: Config,
    stats: Dict[str, float],
) -> Tuple[List[float], torch.Tensor]:
    # block i is verified from drafted anchor x_hat[i]
    x_batch = torch.cat([x_hat[i] for i in range(num_blocks)], dim=0)

    for local_step in range(k_per_block):
        current_positions = torch.tensor(
            [((start_macro_idx + i) * k_per_block) + local_step for i in range(num_blocks)],
            device=x_batch.device,
            dtype=torch.long,
        )
        next_positions = current_positions + 1
        timestep_values = timesteps[current_positions]

        before = stats["unet_calls"]
        eps_batch = predict_noise_batch(
            pipe,
            x_batch,
            timestep_values,
            uncond_embed,
            cond_embed,
            cfg.guidance_scale,
            stats,
        )
        stats["batched_verify_unet_calls"] += stats["unet_calls"] - before

        x_batch = batched_ddim_step_positions(
            pipe.scheduler,
            x_batch,
            eps_batch,
            timesteps,
            current_positions,
            next_positions,
        )

    x_tilde_endpoints = x_batch
    x_hat_endpoints = torch.cat([x_hat[i + 1] for i in range(num_blocks)], dim=0)
    errs = rel_l2_error_batch(x_hat_endpoints, x_tilde_endpoints)
    block_errors = [float(v.item()) for v in errs]
    return block_errors, x_tilde_endpoints


@torch.no_grad()
def run_windowed_parallel_prefix_decode(
    pipe: StableDiffusionPipeline,
    latents0: torch.Tensor,
    uncond_embed: torch.Tensor,
    cond_embed: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, Dict[str, float], List[Dict[str, object]]]:
    pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=cfg.device)
    timesteps = pipe.scheduler.timesteps

    n_total = cfg.num_inference_steps
    m_total = cfg.num_macro_steps
    if n_total % m_total != 0:
        raise ValueError(
            f"num_inference_steps={n_total} must be divisible by num_macro_steps={m_total}"
        )
    k_per_block = n_total // m_total
    window_size = min(cfg.verify_window_size, m_total)

    stats = {
        "unet_calls": 0.0,
        "draft_unet_calls": 0.0,
        "batched_verify_unet_calls": 0.0,
        "accepted_blocks": 0.0,
        "rejected_blocks": 0.0,
        "fallback_exact_blocks": 0.0,
        "num_windows": 0.0,
        "macro_block_size": float(k_per_block),
    }
    window_records: List[Dict[str, object]] = []

    current_macro_idx = 0
    current_latents = latents0.clone()

    maybe_sync(cfg.device)
    t0 = time.perf_counter()

    while current_macro_idx < m_total:
        remaining = m_total - current_macro_idx
        local_blocks = min(window_size, remaining)
        stats["num_windows"] += 1.0

        draft_stats = {"unet_calls": 0.0}
        x_hat = build_window_coarse_trajectory(
            pipe=pipe,
            x_start=current_latents,
            timesteps=timesteps,
            start_macro_idx=current_macro_idx,
            num_blocks=local_blocks,
            k_per_block=k_per_block,
            uncond_embed=uncond_embed,
            cond_embed=cond_embed,
            cfg=cfg,
            stats=draft_stats,
        )
        stats["draft_unet_calls"] += draft_stats["unet_calls"]
        stats["unet_calls"] += draft_stats["unet_calls"]

        verify_before = stats["unet_calls"]
        block_errors, x_tilde_endpoints = verify_window_parallel_local(
            pipe=pipe,
            x_hat=x_hat,
            timesteps=timesteps,
            start_macro_idx=current_macro_idx,
            num_blocks=local_blocks,
            k_per_block=k_per_block,
            uncond_embed=uncond_embed,
            cond_embed=cond_embed,
            cfg=cfg,
            stats=stats,
        )
        verify_calls = stats["unet_calls"] - verify_before

        accepted_prefix = 0
        for err in block_errors:
            if err <= cfg.tau:
                accepted_prefix += 1
            else:
                break

        if accepted_prefix > 0:
            stats["accepted_blocks"] += float(accepted_prefix)
            if accepted_prefix < local_blocks:
                stats["rejected_blocks"] += 1.0
            current_latents = x_tilde_endpoints[accepted_prefix - 1 : accepted_prefix].clone()
            current_macro_idx += accepted_prefix
            advanced_mode = "accepted_prefix"
        else:
            # No drafted block accepted. Still advance one exact block using the
            # verified endpoint of the first block, then restart a new window.
            stats["rejected_blocks"] += 1.0
            stats["fallback_exact_blocks"] += 1.0
            current_latents = x_tilde_endpoints[0:1].clone()
            current_macro_idx += 1
            advanced_mode = "fallback_exact_first_block"

        window_records.append(
            {
                "window_index": int(stats["num_windows"] - 1),
                "start_macro_idx": current_macro_idx - (accepted_prefix if accepted_prefix > 0 else 1),
                "num_blocks_in_window": local_blocks,
                "draft_unet_calls": float(draft_stats["unet_calls"]),
                "verify_unet_calls": float(verify_calls),
                "block_errors": block_errors,
                "accepted_prefix": accepted_prefix,
                "advance_mode": advanced_mode,
                "next_macro_idx": current_macro_idx,
            }
        )

    maybe_sync(cfg.device)
    stats["wall_time_sec"] = float(time.perf_counter() - t0)
    stats["total_committed_blocks"] = float(
        stats["accepted_blocks"] + stats["fallback_exact_blocks"]
    )
    stats["accept_rate_over_attempted_windows"] = float(
        stats["accepted_blocks"] / max(stats["accepted_blocks"] + stats["rejected_blocks"], 1.0)
    )

    return current_latents, stats, window_records


# ============================================================
# Pipeline / Main
# ============================================================

def load_pipe(cfg: Config) -> StableDiffusionPipeline:
    dtype = torch_dtype_from_string(cfg.dtype)
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(cfg.device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def parse_args() -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default=Config.model_id)
    parser.add_argument("--prompt", type=str, default=Config.prompt)
    parser.add_argument("--negative_prompt", type=str, default=Config.negative_prompt)
    parser.add_argument("--output_dir", type=str, default=Config.output_dir)

    parser.add_argument("--height", type=int, default=Config.height)
    parser.add_argument("--width", type=int, default=Config.width)
    parser.add_argument("--num_inference_steps", type=int, default=Config.num_inference_steps)
    parser.add_argument("--num_macro_steps", type=int, default=Config.num_macro_steps)
    parser.add_argument("--verify_window_size", type=int, default=Config.verify_window_size)
    parser.add_argument("--guidance_scale", type=float, default=Config.guidance_scale)
    parser.add_argument("--seed", type=int, default=Config.seed)

    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--dtype", type=str, default=Config.dtype)

    parser.add_argument("--tau", type=float, default=Config.tau)
    parser.add_argument("--save_images", action="store_true")

    args = parser.parse_args()

    return Config(
        model_id=args.model_id,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        num_macro_steps=args.num_macro_steps,
        verify_window_size=args.verify_window_size,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        tau=args.tau,
        save_images=args.save_images,
    )


@torch.no_grad()
def main() -> None:
    cfg = parse_args()
    ensure_dir(cfg.output_dir)

    print("=" * 90)
    print("CONFIG")
    print("=" * 90)
    for k, v in asdict(cfg).items():
        print(f"{k:>28}: {v}")

    pipe = load_pipe(cfg)
    dtype = torch_dtype_from_string(cfg.dtype)

    pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=cfg.device)
    uncond_embed, cond_embed = encode_prompt_pair(
        pipe,
        cfg.prompt,
        cfg.negative_prompt,
        cfg.device,
        dtype,
    )

    generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    latents0 = prepare_latents(pipe, cfg, generator)

    # tiny warmup
    dummy_stats = {"unet_calls": 0.0}
    _ = predict_noise_single(
        pipe,
        latents0,
        pipe.scheduler.timesteps[0].reshape(1),
        uncond_embed,
        cond_embed,
        cfg.guidance_scale,
        dummy_stats,
    )

    baseline_latents, baseline_stats = run_baseline(
        pipe, latents0, uncond_embed, cond_embed, cfg
    )

    final_latents, decode_stats, window_records = run_windowed_parallel_prefix_decode(
        pipe=pipe,
        latents0=latents0,
        uncond_embed=uncond_embed,
        cond_embed=cond_embed,
        cfg=cfg,
    )

    summary = {
        "baseline_unet_calls": baseline_stats["unet_calls"],
        "windowed_draft_unet_calls": decode_stats["draft_unet_calls"],
        "windowed_parallel_verify_unet_calls": decode_stats["batched_verify_unet_calls"],
        "total_windowed_unet_calls": decode_stats["unet_calls"],
        "call_ratio_total_over_baseline": (
            decode_stats["unet_calls"] / max(baseline_stats["unet_calls"], 1.0)
        ),
        "baseline_time_sec": baseline_stats["wall_time_sec"],
        "windowed_total_time_sec": decode_stats["wall_time_sec"],
        "time_ratio_total_over_baseline": (
            decode_stats["wall_time_sec"] / max(baseline_stats["wall_time_sec"], 1e-12)
        ),
        "final_rel_error_vs_baseline": rel_l2_error(final_latents, baseline_latents),
        **decode_stats,
    }

    baseline_path = ""
    final_path = ""
    if cfg.save_images:
        baseline_img = decode_latents_to_pil(pipe, baseline_latents)
        final_img = decode_latents_to_pil(pipe, final_latents)

        baseline_path = os.path.join(cfg.output_dir, "baseline.png")
        final_path = os.path.join(cfg.output_dir, "windowed_parallel_prefix_result.png")

        baseline_img.save(baseline_path)
        final_img.save(final_path)

    stats_path = os.path.join(cfg.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "baseline_stats": baseline_stats,
                "decode_stats": decode_stats,
                "window_records": window_records,
                "summary": summary,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:>38}: {v:.6f}")
        else:
            print(f"{k:>38}: {v}")

    print("\nWINDOW RECORDS")
    for record in window_records:
        print(
            f"window={record['window_index']:02d} "
            f"start_macro={record['start_macro_idx']:02d} "
            f"blocks={record['num_blocks_in_window']} "
            f"accepted_prefix={record['accepted_prefix']} "
            f"advance_mode={record['advance_mode']} "
            f"next_macro={record['next_macro_idx']:02d} "
            f"errors={[round(x, 6) for x in record['block_errors']]}"
        )

    print("\nSaved:")
    if baseline_path:
        print(f"  {baseline_path}")
        print(f"  {final_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
