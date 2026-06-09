############################################################################
## model.py  —  Fashion MNIST Swarm Learning with Adaptive DP + CascadedDP
## Drop-in replacement for your original model.py.
##
## New env vars:
##   DP_AUTO_TUNE=true          → enable auto-tuner (overrides NOISE_MULTIPLIER
##                                and L2_NORM_CLIP if set)
##   DP_TARGET_EPSILON=10.0     → privacy budget the tuner aims for
##   DP_ADAPTIVE_CALLBACK=true  → enable self-healing mid-training adjustments
##
##   CASCADED_DP=true           → enable SNR-based DP drop (CascadedDPCallback)
##   SNR_PLATEAU_EPS=0.02       → drop DP when SNR relative change < this (2% default)
##   DP_DROP_WINDOW=5           → rolling window size for SNR averaging
##   MIN_DP_EPOCHS=5            → minimum epochs before DP drop is evaluated
##   NODE_ID=0                  → this node's integer ID (0-indexed)
##   NUM_NODES=2                → total number of swarm nodes
############################################################################

import os
import json
import time
import numpy as np
import tensorflow as tf
from collections import deque
from sklearn.metrics import f1_score

from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy
)

from swarmlearning.tf import SwarmCallback
from dp_autotuner import DPAutoTuner, DPAdaptiveCallback   # ← new


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

batchSize       = 32
defaultMaxEpoch = 20
defaultMinPeers = 2


# ─────────────────────────────────────────────────────────────────────────────
# DECENTRALIZED CASCADED DP CALLBACK  (SNR-gated, Swarm Quorum)
# ─────────────────────────────────────────────────────────────────────────────

class CascadedDPCallback(tf.keras.callbacks.Callback):
    """
    Drops differential privacy across ALL swarm nodes simultaneously using a
    fully decentralized Quorum-based Peer Consensus mechanism.

    Drop criterion: SNR plateau — the epoch-over-epoch relative change in SNR
    falls below snr_plateau_eps.

        SNR         =  mean_grad_norm  /  noise_std
        Δ_rel(SNR)  =  |SNR_now - SNR_prev| / SNR_prev

    where  noise_std = l2_norm_clip * noise_multiplier / sqrt(batch_size)
    is the effective per-parameter DP noise injected by the DP optimizer.

    Using a relative change rate instead of a fixed threshold makes the
    trigger self-calibrating: it fires when SNR has stopped evolving
    (plateaued — rising or falling), regardless of what the absolute SNR
    value is, so it works across architectures and hyperparameter sets
    without re-tuning.
    """

    def __init__(
        self,
        val_ds,
        node_id,
        num_nodes,
        scratch_dir,
        noise_multiplier,
        l2_norm_clip,
        batch_size,
        optimizer_type='sgd',
        learning_rate=0.01,
        window_size=5,
        snr_plateau_eps=0.02,
        min_dp_epochs=5,
    ):
        super().__init__()

        self.val_ds           = val_ds
        self.node_id          = node_id
        self.num_nodes        = num_nodes
        self.scratch_dir      = scratch_dir
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip     = l2_norm_clip
        self.batch_size       = batch_size
        self.optimizer_type   = optimizer_type
        self.learning_rate    = learning_rate
        self.window_size      = window_size
        self.snr_plateau_eps  = snr_plateau_eps   # relative change threshold (dimensionless)
        self.min_dp_epochs    = min_dp_epochs

        # Pre-compute the effective DP noise standard deviation
        # σ_eff = (l2_norm_clip × noise_multiplier) / sqrt(batch_size)
        self.noise_std = (l2_norm_clip * noise_multiplier) / (batch_size ** 0.5)

        # Tracking windows
        self.grad_norm_window = deque(maxlen=window_size)
        self.snr_window       = deque(maxlen=window_size)

        # Telemetry history
        self.grad_history      = []
        self.snr_history       = []
        self.snr_rel_ch_history = []   # relative change history for diagnostics

        self.dp_active      = True
        self.dp_drop_epoch  = None
        self.dp_drop_reason = None

        # Consensus vote file for this node
        self.vote_file = os.path.join(
            self.scratch_dir, f".vote_drop_dp_node_{self.node_id}"
        )

        # Clean up stale vote files from previous runs
        if os.path.exists(self.vote_file):
            try:
                os.remove(self.vote_file)
            except Exception:
                pass

        self._measure_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_grad_norm(self):
        """Compute stable gradient norm sampled across up to 5 validation batches."""
        norms = []
        for x, y in self.val_ds.take(5):
            with tf.GradientTape() as tape:
                preds    = self.model(x, training=False)
                loss_val = self._measure_loss(y, preds)
            grads     = tape.gradient(loss_val, self.model.trainable_variables)
            grad_norm = tf.linalg.global_norm(grads).numpy()
            norms.append(float(grad_norm))
        return float(np.mean(norms))

    def _drop_dp(self, epoch, snr_rel_change):
        """Swap to a standard (non-DP) optimizer, recompile, and rebuild the runtime graph."""

        print(f"\n***** CascadedDP: [Node {self.node_id}] SWARM QUORUM UNLOCKED *****")
        print(
            f"***** CascadedDP: SNR plateaued (Δ_rel={snr_rel_change:.4f} "
            f"< {self.snr_plateau_eps}) — dropping DP at epoch {epoch + 1} *****"
        )

        if self.optimizer_type == 'adam':
            new_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            new_optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate, momentum=0.9, nesterov=True
            )

        new_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.model.compile(
            loss=new_loss,
            optimizer=new_optimizer,
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
        )

        # Force rebuild of Keras low-level execution graphs
        if hasattr(self.model, 'train_function'):
            self.model.train_function   = None
            self.model.test_function    = None
            self.model.predict_function = None

            for x_sample, y_sample in self.val_ds.take(1):
                self.model.make_train_function()
                self.model.make_test_function()
                self.model.make_predict_function()

        self.dp_active     = False
        self.dp_drop_epoch = epoch + 1

        self.dp_drop_reason = {
            "epoch":            epoch + 1,
            "snr_plateau_eps":  self.snr_plateau_eps,
            "snr_rel_change":   snr_rel_change,
            "rolling_snr":      float(np.mean(self.snr_window)),
            "noise_std":        self.noise_std,
            "snr_history":      self.snr_history[:],
        }

        print(f"***** CascadedDP: low-level execution graphs forcefully rebuilt *****")
        print(f"***** CascadedDP: model recompiled with standard optimizer *****\n")

    # ── Main hook ────────────────────────────────────────────────────────────

    def on_epoch_end(self, epoch, logs=None):
        if not self.dp_active:
            return

        grad_norm = self._compute_grad_norm()

        # SNR = signal (gradient norm) / noise (effective DP noise std)
        snr = grad_norm / self.noise_std if self.noise_std > 0 else float('inf')

        self.grad_norm_window.append(grad_norm)
        self.snr_window.append(snr)

        self.grad_history.append(float(grad_norm))
        self.snr_history.append(float(snr))

        rolling_snr = float(np.mean(self.snr_window))

        # Relative SNR change: |SNR_now - SNR_prev| / SNR_prev
        # Only meaningful once we have at least 2 history points
        if len(self.snr_history) >= 2:
            prev_snr    = self.snr_history[-2]
            snr_rel_change = abs(snr - prev_snr) / prev_snr if prev_snr > 0 else 1.0
        else:
            snr_rel_change = 1.0   # no plateau yet

        self.snr_rel_ch_history.append(float(snr_rel_change))

        print(
            f"  [CascadedDP] Node={self.node_id} | epoch={epoch + 1} | "
            f"grad_norm={grad_norm:.6f} | noise_std={self.noise_std:.6f} | "
            f"SNR={snr:.4f} | rolling_SNR={rolling_snr:.4f} | "
            f"Δ_rel(SNR)={snr_rel_change:.4f} (plateau_eps={self.snr_plateau_eps})"
        )

        # 1. Minimum-epochs guard
        if epoch + 1 < self.min_dp_epochs or len(self.snr_window) < self.window_size:
            return

        # 2. Local plateau check: SNR has stopped changing relative to its own scale
        #    This is dimensionless — works regardless of absolute SNR value or architecture
        if snr_rel_change < self.snr_plateau_eps:
            if not os.path.exists(self.vote_file):
                try:
                    with open(self.vote_file, 'w') as f:
                        f.write(
                            f"Node {self.node_id} voted at epoch {epoch + 1} "
                            f"| Δ_rel(SNR)={snr_rel_change:.4f} < {self.snr_plateau_eps}"
                        )
                    print(
                        f"  [CascadedDP-Consensus] Node {self.node_id} posted drop-DP vote "
                        f"(Δ_rel(SNR)={snr_rel_change:.4f} < eps={self.snr_plateau_eps})."
                    )
                except Exception as e:
                    print(f"  [CascadedDP-Consensus] Error writing vote file: {e}")

        # 3. Scan shared scratch volume for all peer votes
        total_votes = sum(
            1 for peer_id in range(self.num_nodes)
            if os.path.exists(
                os.path.join(self.scratch_dir, f".vote_drop_dp_node_{peer_id}")
            )
        )

        print(
            f"  [CascadedDP-Quorum] Node {self.node_id}: "
            f"{total_votes}/{self.num_nodes} votes collected."
        )

        # 4. Strict quorum: ALL nodes must agree before DP is dropped
        if total_votes == self.num_nodes:
            # Stagger drops slightly to avoid file-system race conditions
            time.sleep(self.node_id * 0.2)
            self._drop_dp(epoch, snr_rel_change)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_optimizer(
    optimizer_type,
    dp_enabled,
    learning_rate,
    l2_norm_clip,
    noise_multiplier,
    microbatches
):
    """Return DP or standard optimizer."""

    if dp_enabled:

        if optimizer_type == 'adam':
            from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
                DPKerasAdamOptimizer
            )
            print("***** Using DP-Adam optimizer")
            return DPKerasAdamOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=learning_rate or 0.001
            )
        else:
            from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
                DPKerasSGDOptimizer
            )
            print("***** Using DP-SGD optimizer")
            return DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=learning_rate or 0.01
            )

    else:

        if optimizer_type == 'adam':
            print("***** Using Adam optimizer")
            return tf.keras.optimizers.Adam(learning_rate=learning_rate or 0.001)
        else:
            print("***** Using SGD optimizer")
            return tf.keras.optimizers.SGD(
                learning_rate=learning_rate or 0.01,
                decay=1e-6, momentum=0.9, nesterov=True
            )


def get_metrics():
    return [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]


def main():

    modelName = 'fashion-mnist'

    # ── Read env vars ──────────────────────────────────────────────────────

    scratchDir   = os.getenv('SCRATCH_DIR', '/platform/scratch')
    maxEpoch     = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
    minPeers     = int(os.getenv('MIN_PEERS', str(defaultMinPeers)))
    dpEnabled    = os.getenv('DP_ENABLED', 'false').lower() == 'true'
    optimizerType = os.getenv('OPTIMIZER', 'sgd').lower()
    learningRate  = float(os.getenv('LEARNING_RATE', '0'))
    nodeId        = int(os.getenv('NODE_ID', '0'))
    numNodes      = int(os.getenv('NUM_NODES', '2'))

    # Auto-tuner flags
    dpAutoTune       = os.getenv('DP_AUTO_TUNE', 'false').lower() == 'true'
    dpTargetEpsilon  = float(os.getenv('DP_TARGET_EPSILON', '10.0'))
    dpAdaptive       = os.getenv('DP_ADAPTIVE_CALLBACK', 'true').lower() == 'true'

    # Manual overrides (used only when auto-tune is OFF)
    noiseMultiplier = float(os.getenv('NOISE_MULTIPLIER', '0.0'))
    l2NormClip      = float(os.getenv('L2_NORM_CLIP', '1.0'))
    microbatches    = int(os.getenv('MICROBATCHES', str(batchSize)))

    # CascadedDP (SNR plateau-gated) flags
    cascadedDp    = os.getenv('CASCADED_DP', 'false').lower() == 'true'
    snrPlateauEps = float(os.getenv('SNR_PLATEAU_EPS', '0.02'))  # relative change threshold
    dpDropWindow  = int(os.getenv('DP_DROP_WINDOW', '5'))
    minDpEpochs   = int(os.getenv('MIN_DP_EPOCHS', '5'))

    os.makedirs(scratchDir, exist_ok=True)

    print('***** Starting model =', modelName)
    print('-' * 64)

    # ── Load data ──────────────────────────────────────────────────────────

    print("Loading Fashion MNIST dataset ..")

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    # ── Split dataset per node ─────────────────────────────────────────────

    total_samples = len(x_train)
    split_size    = total_samples // numNodes
    start         = nodeId * split_size
    end           = total_samples if nodeId == numNodes - 1 else start + split_size

    x_train = x_train[start:end]
    y_train = y_train[start:end]

    print(f'Node ID: {nodeId} | Total Nodes: {numNodes}')
    print(f'Training sample range: {start} → {end}')
    print(f'Local training dataset size: {len(x_train)}')
    print('-' * 64)

    # ── Normalise ──────────────────────────────────────────────────────────

    x_train = x_train / 255.0
    x_test  = x_test  / 255.0

    num_train_samples = len(x_train)

    # ── One-hot encode ─────────────────────────────────────────────────────

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test,  10)

    # ──────────────────────────────────────────────────────────────────────
    # DP AUTO-TUNER  (runs before model build if enabled)
    # ──────────────────────────────────────────────────────────────────────

    tuner_report  = None
    adaptive_cb   = None

    if dpEnabled and dpAutoTune:

        tuner = DPAutoTuner(target_epsilon=dpTargetEpsilon)

        suggested = tuner.suggest(
            x_train     = x_train,
            y_train     = y_train,
            epochs      = maxEpoch,
            batch_size  = batchSize,
            num_nodes   = numNodes,
        )

        # Override the manual env-var values with auto-tuned ones
        noiseMultiplier = suggested["noise_multiplier"]
        l2NormClip      = suggested["l2_norm_clip"]
        microbatches    = suggested["microbatches"]
        tuner_report    = suggested

        print(f"  [AutoTune] noise_multiplier = {noiseMultiplier}")
        print(f"  [AutoTune] l2_norm_clip     = {l2NormClip}")
        print(f"  [AutoTune] estimated ε      = {suggested['estimated_epsilon']}")

        # Prepare the adaptive callback
        if dpAdaptive:
            adaptive_cb = DPAdaptiveCallback(
                initial_noise = noiseMultiplier,
                initial_clip  = l2NormClip,
                log_file      = os.path.join("/results", "dp_adaptive_log.jsonl"),
                verbose       = True,
            )
            print("  [AutoTune] DPAdaptiveCallback enabled — "
                  "noise will self-heal during training.\n")

    # ── Model ──────────────────────────────────────────────────────────────

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64,  activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32,  activation='relu'),
        tf.keras.layers.Dense(10,  activation='softmax'),
    ])

    # ── Optimizer ─────────────────────────────────────────────────────────

    optimizer = get_optimizer(
        optimizerType, dpEnabled,
        learningRate or None,
        l2NormClip, noiseMultiplier, microbatches
    )

    # ── Loss ──────────────────────────────────────────────────────────────

    if dpEnabled:
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    metrics = get_metrics()

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(model.summary())

    # ── Data pipelines ─────────────────────────────────────────────────────

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(num_train_samples)
        .batch(batchSize, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batchSize)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Callbacks ─────────────────────────────────────────────────────────

    swarmCallback = SwarmCallback(
        syncFrequency=1024,
        minPeers=minPeers,
        adsValData=val_ds,
        adsValBatchSize=batchSize,
        mergeMethod='mean',
        totalEpochs=maxEpoch,
    )

    callbacks = [swarmCallback]
    cascadedDpCallback = None

    if adaptive_cb is not None:
        callbacks.append(adaptive_cb)   # ← self-healing DP callback

    if dpEnabled and cascadedDp:
        actual_lr = learningRate or (0.001 if optimizerType == 'adam' else 0.01)
        cascadedDpCallback = CascadedDPCallback(
            val_ds           = val_ds,
            node_id          = nodeId,
            num_nodes        = numNodes,
            scratch_dir      = scratchDir,
            noise_multiplier = noiseMultiplier,
            l2_norm_clip     = l2NormClip,
            batch_size       = batchSize,
            optimizer_type   = optimizerType,
            learning_rate    = actual_lr,
            window_size      = dpDropWindow,
            snr_plateau_eps  = snrPlateauEps,
            min_dp_epochs    = minDpEpochs,
        )
        callbacks.append(cascadedDpCallback)
        print(
            f"  [CascadedDP] Enabled | SNR_PLATEAU_EPS={snrPlateauEps} | "
            f"window={dpDropWindow} | min_epochs={minDpEpochs} | "
            f"noise_std={cascadedDpCallback.noise_std:.6f}"
        )

    # ── Train ──────────────────────────────────────────────────────────────

    print('Starting training ...')
    train_start = time.time()

    model.fit(
        train_ds,
        epochs=maxEpoch,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    train_end     = time.time()
    training_time = round(train_end - train_start, 2)

    print('Training done!')
    print(f"***** Training time: {training_time}s ({round(training_time/60, 2)} min)")

    # ── Collect final DP params (may have been adjusted mid-training) ──────

    final_noise = noiseMultiplier
    final_clip  = l2NormClip
    adaptive_summary = None

    if adaptive_cb is not None:
        adaptive_summary = adaptive_cb.summary()
        final_noise = adaptive_summary["final_noise"]
        final_clip  = adaptive_summary["final_clip"]

        print(f"\n  [DPAdaptiveCallback] Summary:")
        print(f"    Adjustments made : {adaptive_summary['adjustments_made']}")
        print(f"    Final noise      : {final_noise}")
        print(f"    Final clip       : {final_clip}")

    # ── Privacy report ─────────────────────────────────────────────────────

    eps = None

    if dpEnabled and final_noise > 0:

        print('-' * 64)
        print('***** PRIVACY REPORT *****')

        delta = 1.0 / num_train_samples

        # Only count the epochs during which DP was actually active
        dp_epochs = (
            cascadedDpCallback.dp_drop_epoch
            if (cascadedDpCallback and cascadedDpCallback.dp_drop_epoch)
            else maxEpoch
        )

        eps, _ = compute_dp_sgd_privacy(
            n=num_train_samples,
            batch_size=batchSize,
            noise_multiplier=final_noise,
            epochs=dp_epochs,
            delta=delta,
        )

        print(f"DP active for {dp_epochs}/{maxEpoch} epochs")
        print(f"Final Epsilon (ε): {eps:.4f}")
        print(f"Final Delta   (δ): {delta:.2e}")
        print('**************************')
        print('-' * 64)

    elif dpEnabled and final_noise <= 0:
        print("***** WARNING: noise_multiplier is 0.0 — privacy budget is infinite.")

    # ── Evaluate ───────────────────────────────────────────────────────────

    scores     = model.evaluate(val_ds, verbose=1)
    score_names = ['loss'] + [m.name for m in metrics]

    for name, val in zip(score_names, scores):
        print(f"***** Test {name}: {val:.4f}")

    y_pred         = model.predict(val_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test,  axis=1)

    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    print(f"***** Test f1_score: {f1:.4f}")

    # ── Save results ───────────────────────────────────────────────────────

    results = {

        "config": {
            "dp_enabled":       dpEnabled,
            "cascaded_dp":      cascadedDp,
            "auto_tuned":       dpAutoTune,
            "noise_multiplier": final_noise,
            "l2_norm_clip":     final_clip,
            "microbatches":     microbatches,
            "optimizer":        optimizerType,
            "learning_rate":    learningRate or "default",
            "epochs":           maxEpoch,
            "node_id":          nodeId,
            "num_nodes":        numNodes,
        },

        "privacy": {
            "epsilon":          round(eps, 4) if eps is not None else None,
            "delta":            float(1.0 / num_train_samples) if dpEnabled else None,
            "dp_drop_epoch":    cascadedDpCallback.dp_drop_epoch if cascadedDpCallback else None,
            "snr_plateau_eps":  snrPlateauEps if cascadedDp else None,
            "dp_drop_reason":   cascadedDpCallback.dp_drop_reason if cascadedDpCallback else None,
        },

        "results": {
            **{name: round(float(val), 4) for name, val in zip(score_names, scores)},
            "f1_score": round(float(f1), 4),
        },

        "timing": {
            "training_time_seconds": training_time,
            "training_time_minutes": round(training_time / 60, 2),
        },

        # ── Sections added by auto-tuner / cascaded-DP ────────────────
        "auto_tune_report":  tuner_report,
        "adaptive_summary":  adaptive_summary,
    }

    result_file  = os.getenv("RESULT_FILE", "results.json")
    results_path = os.path.join("/results", result_file)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")

    # ── Save model ─────────────────────────────────────────────────────────

    model_path = os.path.join(scratchDir, modelName)
    model.save(model_path)
    print('Saved the trained model!')


if __name__ == '__main__':
    main()
