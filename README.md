<h1 align="center">RESETTLE:<br>Robotic Recovery through Disagreement-Triggered Retrieval and Efficient Corrective Control</h1>

<p align="center">
  <strong>
    Yuxin Chen &nbsp;&middot;&nbsp; Senqiao Yang &nbsp;&middot;&nbsp; Zixuan Wang &nbsp;&middot;&nbsp; Jinhui Ye &nbsp;&middot;&nbsp; Changsheng Lu<br>
    Pengguang Chen &nbsp;&middot;&nbsp; Shu Liu &nbsp;&middot;&nbsp; Zhuotao Tian &nbsp;&middot;&nbsp; Jiaya Jia
  </strong>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2610.12185"><img src="https://img.shields.io/badge/Paper-arXiv%3A2610.12185-b31b1b" alt="Paper: arXiv 2610.12185"></a>
  <a href="#demo"><img src="https://img.shields.io/badge/Video-Demo-2563eb" alt="Demo video"></a>
  <a href="https://github.com/JIA-Lab-research/RESETTLE"><img src="https://img.shields.io/badge/Code-available-brightgreen" alt="Code available"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-6f42c1" alt="BibTeX citation"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> &nbsp;&middot;&nbsp;
  <a href="#demo">Demo</a> &nbsp;&middot;&nbsp;
  <a href="#method">Method</a> &nbsp;&middot;&nbsp;
  <a href="#getting-started">Getting Started</a> &nbsp;&middot;&nbsp;
  <a href="#citation">Citation</a>
</p>

## Demo

https://github.com/user-attachments/assets/195cb46d-e280-44ca-8fac-e4bbd69da30f

<p align="center">
  <a href="https://github.com/JIA-Lab-research/RESETTLE/raw/refs/heads/main/assets/RESETTLE_demo.mp4">Download the full-resolution demo (MP4)</a>
</p>

## Overview

**RESETTLE** is a model-agnostic recovery framework for frozen robot policies. It detects persistent disagreement between independently sampled action proposals, retrieves a same-task demonstration reference, and combines a state-servo prior with a guarded visual residual to execute one corrective action before returning control to the base policy. See the [paper](https://arxiv.org/pdf/2610.12185) for the full method and experiments.

### Highlights

- **Disagreement-triggered intervention.** Monitor two base-policy action proposals under identical conditioning and intervene when calibrated disagreement persists across decisions.
- **Demonstration-guided recovery.** Retrieve a same-task reference with an adapted V-JEPA encoder and use its image and state to guide correction.
- **Efficient corrective control.** Execute a single corrective action without online trajectory optimization or additional vision-language reasoning, then resume the base policy.

> **Repository scope:** This release provides the **LIBERO training pipeline, expert goal-bank encoding, HTTP recovery service, and guarded executor example** for single-tick (`K=1`) state-servo plus visual-residual recovery.

## Method

<p align="center">
  <img src="assets/method.png" alt="RESETTLE framework: disagreement trigger, V-JEPA retrieval and reference-guided recovery" width="95%">
</p>

*Paper method overview.* Two independently sampled **base-policy** action chunks under identical conditioning are compared at successive decisions; a persistent, calibrated disagreement triggers same-task demonstration retrieval. The retrieved image and state guide a state-servo prior and visual correction for one native action, after which the base policy resumes. This repository covers the **LIBERO training, expert bank and recovery server** shown on the right.

### Why action disagreement?

<p align="center">
  <img src="assets/action_disagreement.png" alt="QwenPI action-disagreement distribution on LIBERO-Plus under layout and robot-initialization perturbations" width="85%">
</p>

*Paper motivation (QwenPI on LIBERO-Plus).* The plotted training-data P95 and the heavier failure tails motivate a two-consecutive-decision trigger. This is a paper result, **not** a threshold estimated by this package. [TWO_SEED_DISAGREEMENT.md](TWO_SEED_DISAGREEMENT.md) instead demonstrates sampling the **recovery head** twice at one fixed observation to inspect its residual variance; those samples must not be substituted for the paper's two independently sampled base-policy proposals or its temporal trigger.

## Getting Started

Run the following commands from the repository root.

### Requirements

Use **Python 3.12** and a **CUDA-capable PyTorch** environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

ViT-G needs substantial GPU memory. The entry points use the following GPU counts:

| Stage | GPUs |
| --- | ---: |
| Training (default) | 8 |
| Expert-bank encoding | 4 |
| Recovery inference | 1 |

Supply a compatible **V-JEPA2-AC ViT-G initial checkpoint** and the four **LIBERO LeRobot expert suites** beneath one directory:

```text
/path/to/libero/
├── libero_object_no_noops_1.0.0_lerobot/
├── libero_goal_no_noops_1.0.0_lerobot/
├── libero_spatial_no_noops_1.0.0_lerobot/
└── libero_10_no_noops_1.0.0_lerobot/
```

Each suite must contain `meta/`, `data/`, and `videos/`. The package does not download models or data.

### 1. Train

```bash
export PYTHON=/path/to/python
export LIBERO_DATA_ROOT=/path/to/libero
export VJEPA_PRETRAIN=/path/to/vjepa2-ac-vitg.pt
export RESETTLE_OUTPUT=/path/to/new_output_directory
export NUM_GPUS=8
bash train.sh
```

[configs/libero.yaml](configs/libero.yaml) specifies four suites, 8 frames at 4 FPS, 256px, a goal at frame 4, `K=1`, `guarded_native_steps=5`, the Gaussian residual head and joint context-encoder/predictor training with dynamics, native-action, gate and saturation losses. [train.sh](train.sh) fills a temporary YAML with the environment's absolute data/weight/output paths. It writes `latest.pt` and `params-pretrain.yaml` under `RESETTLE_OUTPUT`; start in an empty output directory. This is not policy-only fine-tuning.

### 2. Encode Expert Goals

Use the **trained** `latest.pt` for both the bank and the server (not the initial `VJEPA_PRETRAIN` checkpoint). The four-GPU entry assigns one suite per GPU, writes shards, merges them and requires 40 task IDs:

```bash
export RESETTLE_CHECKPOINT="$RESETTLE_OUTPUT/latest.pt"
export RESETTLE_BANK=/path/to/new_bank_directory
export BANK_BATCH_SIZE=16
bash build_bank.sh
```

This produces `success_id_bank.npz` (`features [N,D]`, `states [N,7]`), `success_id_bank.jsonl` and `success_id_bank_summary.json`. Expert video paths in JSONL remain external and must still exist when the server retrieves a goal. The four shards record a SHA-256 of the same trained checkpoint; merge refuses mixed checkpoints or missing tasks. The original four-suite builder is the reference for this portable entry.

### 3. Start the Recovery Server

```bash
bash serve.sh --device cuda:0 --host 127.0.0.1 --port 18082
```

The launcher uses `RESETTLE_CHECKPOINT` and `RESETTLE_BANK` above. The only planning route is `POST /plan-recovery-state-servo-residual`; `GET /health` reports checkpoint/bank metadata and the checkpoint-verified native calibration. The POST JSON needs `task` (LIBERO task text), `image` (RGB uint8 `[H,W,3]`) and `state` (float32 `[7]`: XYZ metres, Euler XYZ radians, gripper closedness). Send each array as `{"format":"npy_base64","data":"<base64 of NumPy .npy bytes>"}`. Optional `goal_reference: {"row": N}` pins a bank row; `retrieval_top_k` controls reported nearest candidates.

The response contains `goal_reference`, `goal_state`, `state_servo_prior [7]`, `jepa_residual_action_chunk [1,7]`, `action_log_std_chunk [1,7]` and `jepa_diff_action_chunk [1,7]`. The last field is the **mean proposed action** (prior plus mean residual), **not** a post-guard executed action. HTTP input/output and sampling usage are demonstrated in [TWO_SEED_DISAGREEMENT.md](TWO_SEED_DISAGREEMENT.md).

## Guarded Native Action

The server deliberately stops before physical execution. [GUARDED_EXECUTOR.md](GUARDED_EXECUTOR.md) and [guarded_executor.py](guarded_executor.py) show the LIBERO-only downstream single-tick conversion: use the server's `/health` calibration, convert the full state-servo delta over five native steps and the masked motion residual over one step, gate XYZ and rotation independently, then add and clip or fall back to the servo baseline. The gripper always follows the baseline. A caller must supply the measured current state, trigger/schedule interventions externally and execute at most one tick before re-observation; this example has no robot or simulator connection.

## Contract and Limits

| Boundary | Compatibility condition |
| --- | --- |
| Training to checkpoint | `training_contract_version=asf1_guarded_s1_native_residual_k1_v7`, state-servo residual, conditioned prior, executed correction, `K=1`, goal frame 4 and 5 native steps. |
| Checkpoint to expert bank | Same trained encoder weights, deterministic 256px transform, mean pooled layer-normalized frame tokens with L2 normalization for nearest-neighbour retrieval; bank and server require the same SHA-256. |
| Checkpoint to server | Strict encoder/predictor/head state-dict loads; server rejects other action contracts and mismatched native calibration. Current and goal frame encoding uses the same encoder as the bank. |
| Prior and action | Both training and server use SO(3) fractional state-servo `D/5` and a learned K=1 residual. State is LIBERO's 7D XYZ/Euler/closedness convention, not raw LeRobot 8D observation state. |

Training applies image augmentation to 8-frame clips; bank and server encode individual frames as repeated tubelets with deterministic preprocessing. This is the original inference convention, **not** proof of identical train/inference feature distributions. Guard/native losses during training encourage suitable corrections, but the server does not execute the downstream native command clipping, direction or ratio guard. The optional [guarded executor example](GUARDED_EXECUTOR.md) computes these from the verified `/health` calibration; it does not trigger interventions or step a robot. Do not treat the raw HTTP action or the two-seed samples as safe to apply directly. Two seeds sample one K=1 Gaussian residual at a fixed observation/goal, not a temporal two-step chunk and not independently trained policies.

## Citation

If you find RESETTLE useful in your research, please cite our [paper](https://arxiv.org/abs/2610.12185):

```bibtex
@article{chen2026resettle,
  title={RESETTLE: Robotic Recovery through Disagreement-Triggered Retrieval and Efficient Corrective Control},
  author={Chen, Yuxin and Yang, Senqiao and Wang, Zixuan and Ye, Jinhui and Lu, Changsheng and Chen, Pengguang and Liu, Shu and Tian, Zhuotao and Jia, Jiaya},
  journal={arXiv preprint arXiv:2610.12185},
  year={2026}
}
```
