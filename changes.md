# ARC Battery Digital Twin Pipeline Refactor Specification

You are refactoring the entire ARC battery thermal digital twin training pipeline for maximum performance, robustness, observability, and fault tolerance.

The hardware target is:

* GPU: NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)
* CPU: Intel i7
* RAM: 16 GB DDR5
* OS: Windows 11
* CUDA: 12.9
* PyTorch: CUDA-enabled

The goal is to maximize GPU utilization while ensuring safe recovery from failures.

---

# PRIORITY 1: PERFORMANCE OPTIMIZATION

## Objective

Maximize utilization of the RTX 4060 and minimize CPU-GPU synchronization overhead.

Current bottlenecks:

* Millions of tiny GPU calls during rollout
* Frequent CPU ↔ GPU transfers
* Tensor allocation inside loops
* Use of `.cpu().item()` during rollout
* NumPy preprocessing inside timestep loops

---

## Training Optimizations

Implement the following:

### 1. Use non-blocking transfers

Replace:

```python
batch_X = batch_X.to(device)
batch_y = batch_y.to(device)
```

with:

```python
batch_X = batch_X.to(device, non_blocking=True)
batch_y = batch_y.to(device, non_blocking=True)
```

Requirements:

* pin_memory=True must remain enabled.
* Use persistent_workers=True.

---

### 2. Replace tensor construction

Replace:

```python
torch.tensor(array)
```

with:

```python
torch.from_numpy(array)
```

Avoid unnecessary memory copies.

---

### 3. Improve GPU memory utilization

Use:

```python
torch.set_float32_matmul_precision("high")
```

Enable:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

Use AMP everywhere:

```python
with torch.amp.autocast("cuda"):
```

---

### 4. Optimize DataLoader

Benchmark and auto-tune:

```python
num_workers = [4, 6, 8]
prefetch_factor = [2, 4]
persistent_workers = True
pin_memory = True
```

Log throughput for each configuration.

---

### 5. Dataset optimization

Replace:

```python
np.searchsorted()
```

with a precomputed mapping:

```python
idx_to_file
```

to achieve O(1) lookup.

Use:

```python
torch.from_numpy()
```

instead of constructing tensors repeatedly.

---

### 6. VRAM caching

Where possible:

* Cache scaler tensors directly on GPU.
* Keep frequently used constants on GPU.

Example:

```python
mean_X_gpu
std_X_gpu
mean_y_gpu
std_y_gpu
```

Avoid repeated CPU-GPU transfers.

---

# PRIORITY 2: ROLLOUT OPTIMIZATION

Current rollout is extremely slow.

The rollout currently performs approximately 13 million forward passes.

Refactor rollout.

---

## Remove synchronization bottlenecks

Replace:

```python
pred_scaled = model(x_tensor).cpu().item()
```

with fully GPU-resident computations.

Never call:

```python
.item()
.cpu()
.numpy()
```

inside timestep loops.

Only move results to CPU at the end.

---

## GPU rollout

Move all of:

* T_pred
* scaler means
* scaler stds
* inputs

onto GPU.

Perform rollout entirely on GPU.

Minimize CPU interaction.

---

## Progress reporting

Print:

```python
Rollout file 3/87
```

and:

```python
Step 50000/150000
```

for large trajectories.

Display ETA.

---

# PRIORITY 3: LOGGING & OBSERVABILITY

I never want silent execution again.

Implement structured logging using Python logging module.

Create:

```text
logs/
    training.log
    rollout.log
    evaluation.log
```

Log both to:

* terminal
* file

Use timestamps.

Format:

```text
2026-06-17 22:10:45 INFO Epoch 18 started
```

---

## Log everything

Training:

* epoch start/end
* epoch duration
* train duration
* validation duration
* batch throughput
* samples/sec
* GPU utilization
* GPU memory allocated
* GPU memory reserved
* CPU usage
* RAM usage

Rollout:

* current file
* timestep progress
* ETA
* duration per file
* total duration

Evaluation:

* start time
* finish time
* metrics

---

## GPU metrics

Log:

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
```

If available, use pynvml or nvidia-ml-py to log:

* GPU utilization
* power draw
* temperature

---

# PRIORITY 4: SAFE EXIT & RECOVERY

No future run should lose progress.

Implement checkpointing.

---

## Save immediately on validation improvement

When validation improves:

```python
torch.save(...)
```

Save:

```text
results/checkpoints/
    best_model.pt
```

Include:

* model state
* optimizer state
* scaler state
* epoch number
* best loss
* config
* random seed

---

## Periodic checkpoints

Every N epochs:

```text
checkpoint_epoch_10.pt
checkpoint_epoch_20.pt
```

---

## Save immediately after training

Refactor:

```python
preprocess()
train()
evaluate()
rollout()
save()
```

into:

```python
preprocess()
train()
save_checkpoint()
evaluate()
rollout()
plots()
save()
```

Training results must survive rollout crashes.

---

## Resume training

Add CLI support:

```bash
python run_baseline.py --resume results/checkpoints/best_model.pt
```

Resume:

* model
* optimizer
* scheduler
* scaler
* epoch count

---

## Graceful interruption

Handle:

```python
KeyboardInterrupt
```

Upon interruption:

1. Save emergency checkpoint.
2. Save logs.
3. Exit gracefully.

Example:

```text
Emergency checkpoint saved:
results/checkpoints/interrupted_epoch_18.pt
```

---

## Exception handling

Wrap major phases:

```python
preprocess()
train()
evaluate()
rollout()
plots()
save()
```

with robust try/except blocks.

Save partial results whenever possible.

Never lose trained weights.

---

# PRIORITY 5: SEPARATE PIPELINES

Split execution into separate scripts:

```text
run_baseline.py
evaluate_baseline.py
rollout_baseline.py
plot_baseline.py
```

This allows rerunning evaluation without retraining.

---

# ACCEPTANCE CRITERIA

The refactor is complete only if:

1. GPU utilization improves significantly.
2. Rollout is faster and visibly reports progress.
3. Logs are saved to files.
4. Training can resume from checkpoints.
5. Interrupting execution never loses progress.
6. Evaluation and rollout can be rerun independently.
7. Every major step reports duration and ETA.
8. No future training run can lose the best model.

Generate the full implementation and modify all affected files accordingly.
