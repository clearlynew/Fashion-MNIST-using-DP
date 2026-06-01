############################################################################
## dp_autotuner.py
## Adaptive Differential Privacy parameter tuner for Swarm Learning.
##
## Provides:
##   - DPAutoTuner   : analyses dataset stats and suggests initial DP params
##   - DPAdaptiveCallback : Keras callback that self-heals params mid-training
############################################################################

import math
import numpy as np
import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — tuning knobs you can adjust
# ─────────────────────────────────────────────────────────────────────────────

# Accuracy thresholds that trigger noise adjustments
ACCURACY_TOO_LOW   = 0.55   # below this → reduce noise (prioritise accuracy)
ACCURACY_GOOD      = 0.80   # above this → can afford to increase noise a bit

# How aggressively to adjust noise each epoch
NOISE_DECREASE_FACTOR = 0.90   # multiply noise by this when accuracy is low
NOISE_INCREASE_FACTOR = 1.05   # multiply noise by this when accuracy is high

# Hard limits so we never go off the rails
NOISE_MIN = 0.3
NOISE_MAX = 2.5

# Clip bounds
CLIP_MIN  = 0.5
CLIP_MAX  = 3.0

# How many warm-up epochs before the callback starts touching anything
WARMUP_EPOCHS = 3


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def profile_dataset(x_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Compute dataset statistics used to guide initial DP parameter selection.

    Returns a dict with:
      n_samples      – number of local training samples
      n_classes      – number of unique label classes
      class_balance  – std of per-class counts (0 = perfectly balanced)
      pixel_variance – mean per-pixel variance (proxy for data complexity)
      input_norm     – mean L2 norm of flattened samples (guides l2_norm_clip)
    """

    n_samples = len(x_train)

    # Handle one-hot or integer labels
    if y_train.ndim > 1:
        y_int = np.argmax(y_train, axis=1)
    else:
        y_int = y_train.astype(int)

    n_classes = len(np.unique(y_int))

    counts       = np.bincount(y_int, minlength=n_classes).astype(float)
    class_balance = float(np.std(counts) / (np.mean(counts) + 1e-8))

    # Flatten images to compute gradient-proxy norms
    flat = x_train.reshape(n_samples, -1).astype(np.float32)
    norms = np.linalg.norm(flat, axis=1)

    pixel_variance = float(np.var(flat))
    input_norm_p75 = float(np.percentile(norms, 75))  # 75th pct is a safe clip value

    return {
        "n_samples":      n_samples,
        "n_classes":      n_classes,
        "class_balance":  round(class_balance, 4),
        "pixel_variance": round(pixel_variance, 6),
        "input_norm_p75": round(input_norm_p75, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DP AUTO-TUNER
# ─────────────────────────────────────────────────────────────────────────────

class DPAutoTuner:
    """
    Suggests initial noise_multiplier and l2_norm_clip for a given dataset
    and swarm configuration, aiming to balance accuracy vs privacy.

    Usage
    -----
        tuner = DPAutoTuner(target_epsilon=10.0)
        params = tuner.suggest(x_train, y_train,
                               epochs=20, batch_size=32, num_nodes=2)
        print(params)
        # {'noise_multiplier': 1.1, 'l2_norm_clip': 1.2,
        #  'microbatches': 32, 'estimated_epsilon': 9.8, 'profile': {...}}
    """

    def __init__(
        self,
        target_epsilon: float = 10.0,
        delta_scale: float    = 1.0,   # multiplier on 1/n for delta
    ):
        self.target_epsilon = target_epsilon
        self.delta_scale    = delta_scale

    # ------------------------------------------------------------------

    def suggest(
        self,
        x_train:    np.ndarray,
        y_train:    np.ndarray,
        epochs:     int,
        batch_size: int,
        num_nodes:  int = 1,
    ) -> dict:
        """
        Analyse the dataset and return recommended DP parameters.
        """

        profile = profile_dataset(x_train, y_train)
        n       = profile["n_samples"]

        print("\n" + "=" * 64)
        print("  DP AUTO-TUNER — Dataset Profile")
        print("=" * 64)
        print(f"  Local samples      : {n}")
        print(f"  Classes            : {profile['n_classes']}")
        print(f"  Class imbalance σ  : {profile['class_balance']:.4f}"
              f"  {'⚠ imbalanced' if profile['class_balance'] > 0.3 else '✓ balanced'}")
        print(f"  Pixel variance     : {profile['pixel_variance']:.6f}")
        print(f"  Input norm (p75)   : {profile['input_norm_p75']:.4f}")
        print(f"  Swarm nodes        : {num_nodes}")
        print("=" * 64)

        # ── l2_norm_clip ────────────────────────────────────────────────
        # Anchor to the 75th-percentile gradient norm of the flattened input.
        # Clamp to sensible range.
        raw_clip = profile["input_norm_p75"] * 0.1   # scale down from raw pixel norms
        l2_norm_clip = float(np.clip(raw_clip, CLIP_MIN, CLIP_MAX))

        # ── noise_multiplier via binary search ──────────────────────────
        # Find the smallest noise_multiplier whose estimated epsilon
        # is ≤ target_epsilon.
        noise_multiplier = self._search_noise(
            n, batch_size, epochs, l2_norm_clip
        )

        # ── microbatches ─────────────────────────────────────────────────
        # Keep equal to batch_size (TF-Privacy requirement for per-sample grads)
        microbatches = batch_size

        # ── estimated epsilon ────────────────────────────────────────────
        delta = self.delta_scale / n
        eps   = self._estimate_epsilon(
            n, batch_size, noise_multiplier, epochs, delta
        )

        print(f"\n  Suggested parameters")
        print(f"  ─────────────────────────────────────────")
        print(f"  noise_multiplier : {noise_multiplier:.4f}")
        print(f"  l2_norm_clip     : {l2_norm_clip:.4f}")
        print(f"  microbatches     : {microbatches}")
        print(f"  Estimated ε      : {eps:.4f}  (target ≤ {self.target_epsilon})")
        print(f"  δ                : {delta:.2e}")
        print("=" * 64 + "\n")

        return {
            "noise_multiplier":    round(noise_multiplier, 4),
            "l2_norm_clip":        round(l2_norm_clip, 4),
            "microbatches":        microbatches,
            "estimated_epsilon":   round(eps, 4),
            "delta":               delta,
            "profile":             profile,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_epsilon(self, n, batch_size, noise_multiplier, epochs, delta):
        try:
            from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
                compute_dp_sgd_privacy
            )
            eps, _ = compute_dp_sgd_privacy(
                n=n,
                batch_size=batch_size,
                noise_multiplier=noise_multiplier,
                epochs=epochs,
                delta=delta,
            )
            return float(eps)
        except Exception:
            # Rough Gaussian mechanism fallback if TFP not available
            q = batch_size / n
            steps = epochs * (n // batch_size)
            return float(q * math.sqrt(steps * 2 * math.log(1.25 / delta)) / noise_multiplier)

    def _search_noise(self, n, batch_size, epochs, l2_norm_clip):
        """
        Binary search for smallest noise_multiplier whose ε ≤ target_epsilon.
        Falls back to a safe default if the search fails.
        """
        delta = self.delta_scale / n
        lo, hi = NOISE_MIN, NOISE_MAX

        for _ in range(40):
            mid = (lo + hi) / 2
            eps = self._estimate_epsilon(n, batch_size, mid, epochs, delta)
            if eps <= self.target_epsilon:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-4:
                break

        result = hi

        # If even max noise can't hit target, warn and return max
        eps_at_max = self._estimate_epsilon(n, batch_size, NOISE_MAX, epochs, delta)
        if eps_at_max > self.target_epsilon:
            print(f"  ⚠ WARNING: Cannot reach ε={self.target_epsilon} even at "
                  f"noise={NOISE_MAX:.2f} (ε={eps_at_max:.2f}). "
                  f"Consider increasing epochs or target_epsilon.")
            result = NOISE_MAX

        return float(np.clip(result, NOISE_MIN, NOISE_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# SELF-HEALING KERAS CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class DPAdaptiveCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that monitors validation accuracy each epoch and
    adaptively adjusts noise_multiplier and l2_norm_clip on the DP optimizer.

    The "self-healing" logic:
      - If val_accuracy < ACCURACY_TOO_LOW  → reduce noise (help the model learn)
      - If val_accuracy > ACCURACY_GOOD     → can afford slightly more noise
      - Otherwise                           → leave params alone

    Parameters
    ----------
    initial_noise   : starting noise_multiplier (from DPAutoTuner)
    initial_clip    : starting l2_norm_clip
    warmup_epochs   : epochs to wait before making any adjustments
    log_file        : optional path to write per-epoch adjustment log (JSON lines)
    verbose         : print adjustment events
    """

    def __init__(
        self,
        initial_noise:  float,
        initial_clip:   float,
        warmup_epochs:  int   = WARMUP_EPOCHS,
        log_file:       str   = None,
        verbose:        bool  = True,
    ):
        super().__init__()
        self.noise        = initial_noise
        self.clip         = initial_clip
        self.warmup       = warmup_epochs
        self.log_file     = log_file
        self.verbose      = verbose
        self.history      = []   # list of per-epoch dicts

    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_acc = logs.get('val_accuracy', logs.get('val_categorical_accuracy'))

        record = {
            "epoch":           epoch + 1,
            "val_accuracy":    round(float(val_acc), 4) if val_acc is not None else None,
            "noise_multiplier": round(self.noise, 4),
            "l2_norm_clip":    round(self.clip, 4),
            "action":          "none",
        }

        # Don't touch anything during warm-up
        if epoch < self.warmup or val_acc is None:
            self.history.append(record)
            return

        action = "none"

        if val_acc < ACCURACY_TOO_LOW:
            # Model struggling — reduce noise to help it learn
            new_noise = max(self.noise * NOISE_DECREASE_FACTOR, NOISE_MIN)
            if new_noise < self.noise:
                action = f"↓ noise {self.noise:.4f} → {new_noise:.4f}  (val_acc {val_acc:.4f} < {ACCURACY_TOO_LOW})"
                self.noise = new_noise
                self._apply_noise(self.noise)

        elif val_acc > ACCURACY_GOOD:
            # Model doing well — nudge noise up for more privacy
            new_noise = min(self.noise * NOISE_INCREASE_FACTOR, NOISE_MAX)
            if new_noise > self.noise:
                action = f"↑ noise {self.noise:.4f} → {new_noise:.4f}  (val_acc {val_acc:.4f} > {ACCURACY_GOOD})"
                self.noise = new_noise
                self._apply_noise(self.noise)

        if action != "none" and self.verbose:
            print(f"\n  [DPAdaptiveCallback] Epoch {epoch+1}: {action}")

        record["noise_multiplier"] = round(self.noise, 4)
        record["action"]           = action
        self.history.append(record)

        # Optionally persist log
        if self.log_file:
            import json
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------

    def _apply_noise(self, new_noise: float):
        """
        Write new noise_multiplier into the optimizer.
        TF-Privacy DP optimizers expose _noise_multiplier as an attribute.
        """
        opt = self.model.optimizer
        if hasattr(opt, '_noise_multiplier'):
            opt._noise_multiplier = new_noise
        elif hasattr(opt, 'noise_multiplier'):
            opt.noise_multiplier = new_noise
        else:
            print("  [DPAdaptiveCallback] ⚠ Could not update optimizer noise — "
                  "attribute not found. Check TF-Privacy version.")

    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary of adjustments made during training."""
        adjustments = [r for r in self.history if r["action"] != "none"]
        return {
            "total_epochs":      len(self.history),
            "adjustments_made":  len(adjustments),
            "final_noise":       round(self.noise, 4),
            "final_clip":        round(self.clip, 4),
            "adjustment_log":    adjustments,
        }
