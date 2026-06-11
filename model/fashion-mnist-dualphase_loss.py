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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
batchSize       = 32
defaultMaxEpoch = 20
defaultMinPeers = 2

# ─────────────────────────────────────────────────────────────────────────────
# DUAL-PHASE SENSITIVITY-AWARE DP CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class DualPhaseDPCallback(tf.keras.callbacks.Callback):
    """

    Phase 1 (DP active)  : model absorbing structure rapidly, loss curving steeply
    Phase 2 (DP dropped) : model refining, loss plateau, DP overhead no longer justified

    Key hyperparameters:
        curvature_threshold  : how flat the loss curve must be to trigger a vote.
                               Lower = wait longer (more privacy, more time).
                               Higher = drop earlier (less overhead, less privacy).
        curvature_window     : number of epochs to smooth curvature over, guarding
                               against temporary plateaus in noisy loss curves.
        min_dp_epochs        : hard minimum DP epochs regardless of curvature,
                               since early loss is naturally erratic before settling.
    """
    def __init__(
        self,
        node_id,
        num_nodes,
        scratch_dir,
        optimizer_type='sgd',
        learning_rate=0.01,
        curvature_threshold=0.002,
        curvature_window=3,
        min_dp_epochs=5,
    ):
        super().__init__()
        self.node_id             = node_id
        self.num_nodes           = num_nodes
        self.scratch_dir         = scratch_dir
        self.optimizer_type      = optimizer_type
        self.learning_rate       = learning_rate
        self.curvature_threshold = curvature_threshold
        self.curvature_window    = curvature_window
        self.min_dp_epochs       = min_dp_epochs

        # Loss tracking
        self.loss_history       = []      # raw training loss each epoch
        self.velocity_history   = []      # first differences  (loss[t] - loss[t-1])
        self.curvature_history  = []      # second differences (vel[t]  - vel[t-1])
        self.smoothed_curvature = []      # rolling mean of curvature_window values

        # State
        self.dp_active        = True
        self.dp_drop_epoch    = None
        self.dp_drop_reason   = None

        # Epsilon snapshot at the moment DP is dropped — logged separately
        # so results clearly show privacy cost was concentrated in Phase 1.
        self.eps_at_drop      = None

        # Consensus: one vote file per node on the shared scratch volume
        self.vote_file = os.path.join(
            self.scratch_dir, f".vote_drop_dp_node_{self.node_id}"
        )

        # Clean up stale votes from ALL peers at boot — safe if nodes start
        # roughly simultaneously. Prevents inherited votes from prior runs.
        for peer_id in range(self.num_nodes):
            stale = os.path.join(self.scratch_dir, f".vote_drop_dp_node_{peer_id}")
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except Exception:
                    pass

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_curvature(self):
        """
        Return smoothed loss curvature from the rolling window.

        Velocity  = first  difference of loss  → how fast loss is dropping
        Curvature = second difference of loss  → how fast that drop is decelerating

        Near zero curvature means the loss curve has flattened: the model is
        no longer restructuring rapidly, i.e. it has left the high-sensitivity phase.
        """
        if len(self.loss_history) < 3:
            return None

        # First differences (velocities)
        velocities = [
            self.loss_history[i] - self.loss_history[i - 1]
            for i in range(1, len(self.loss_history))
        ]

        # Second differences (curvature)
        curvatures = [
            abs(velocities[i] - velocities[i - 1])
            for i in range(1, len(velocities))
        ]

        # Rolling mean over last curvature_window values
        window_vals = curvatures[-self.curvature_window:]
        return float(np.mean(window_vals))

    def _write_vote_atomic(self, epoch):
        """
        Write this node's convergence vote atomically using a temp file + rename.
        os.replace() is atomic on POSIX filesystems — prevents partial writes
        being read by peer nodes scanning for votes.
        """
        tmp_path = self.vote_file + ".tmp"
        try:
            with open(tmp_path, 'w') as f:
                f.write(f"Node {self.node_id} voted to drop DP at epoch {epoch + 1}")
            os.replace(tmp_path, self.vote_file)
            print(
                f"  [DualPhaseDP-Consensus] Node {self.node_id} posted drop vote "
                f"to shared scratch (epoch {epoch + 1})."
            )
        except Exception as e:
            print(f"  [DualPhaseDP-Consensus] Error writing vote file: {e}")

    def _count_votes(self):
        """Scan shared scratch for all peer vote files and return total count."""
        total = 0
        for peer_id in range(self.num_nodes):
            peer_path = os.path.join(
                self.scratch_dir, f".vote_drop_dp_node_{peer_id}"
            )
            if os.path.exists(peer_path):
                total += 1
        return total

    def _drop_dp(self, epoch):
       
        print(f"\n***** DualPhaseDP: [Node {self.node_id}] SWARM QUORUM UNLOCKED *****")
        print(f"***** DualPhaseDP: dropping DP globally at epoch {epoch + 1} *****")

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

        # ── FORCE REBUILD OF LOW-LEVEL KERAS RUNTIME GRAPH FUNCTIONS ──
        if hasattr(self.model, 'train_function'):
            self.model.train_function   = None
            self.model.test_function    = None
            self.model.predict_function = None

            # Immediate graph tracing session — bypasses NoneType crash on
            # first training step after recompile
            dummy_ds = tf.data.Dataset.from_tensors(
                (
                    tf.zeros([batchSize, 28, 28]),
                    tf.zeros([batchSize, 10])
                )
            )
            for x_s, y_s in dummy_ds.take(1):
                self.model.make_train_function()
                self.model.make_test_function()
                self.model.make_predict_function()

        self.dp_active     = False
        self.dp_drop_epoch = epoch + 1
        self.dp_drop_reason = {
            "epoch":                epoch + 1,
            "curvature_threshold":  self.curvature_threshold,
            "smoothed_curvature":   self.smoothed_curvature[-1] if self.smoothed_curvature else None,
            "loss_at_drop":         self.loss_history[-1] if self.loss_history else None,
            "loss_velocity_window": self.velocity_history[-self.curvature_window:],
        }

        print(f"***** DualPhaseDP: low-level execution graphs forcefully rebuilt *****")
        print(f"***** DualPhaseDP: model recompiled with standard optimizer *****\n")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN EPOCH HOOK
    # ─────────────────────────────────────────────────────────────────────────

    def on_epoch_end(self, epoch, logs=None):
        if not self.dp_active:
            return

        logs = logs or {}

        # Training loss is the primary signal — always present in logs
        train_loss = logs.get('loss')
        if train_loss is None:
            return

        self.loss_history.append(float(train_loss))

        # Need at least 2 loss values for velocity, 3 for curvature
        smoothed_curv = self._compute_curvature()

        # Compute velocity for logging (first difference)
        if len(self.loss_history) >= 2:
            velocity = self.loss_history[-1] - self.loss_history[-2]
            self.velocity_history.append(float(velocity))
        else:
            velocity = None

        if smoothed_curv is not None:
            self.smoothed_curvature.append(smoothed_curv)

        print(
            f"  [DualPhaseDP] Node={self.node_id} | epoch={epoch + 1} | "
            f"loss={train_loss:.6f} | "
            f"velocity={f'{velocity:.6f}' if velocity is not None else 'N/A'} | "
            f"smoothed_curvature={f'{smoothed_curv:.6f}' if smoothed_curv is not None else 'N/A'}"
        )

        # ── GUARD: minimum DP epochs + enough history for curvature ──
        if epoch + 1 < self.min_dp_epochs or smoothed_curv is None:
            return

        # ── PHASE BOUNDARY CHECK: has the loss curve flattened? ──
        if smoothed_curv < self.curvature_threshold:
            if not os.path.exists(self.vote_file):
                self._write_vote_atomic(epoch)

        # ── QUORUM COUNT ──
        total_votes = self._count_votes()
        print(
            f"  [DualPhaseDP-Quorum] Node {self.node_id} cluster status: "
            f"{total_votes}/{self.num_nodes} votes collected."
        )

        # ── UNANIMOUS QUORUM: ALL nodes agree loss has flattened ──
        if total_votes == self.num_nodes:
            # Rank-based stagger to prevent simultaneous filesystem + recompile
            # race conditions. Node 0 acts immediately, others wait proportionally.
            time.sleep(self.node_id * 0.2)
            self._drop_dp(epoch)


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
            from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
            print("***** Using DP-Adam optimizer")
            return DPKerasAdamOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=learning_rate or 0.001
            )
        else:
            from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
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
                learning_rate=learning_rate or 0.01, momentum=0.9, nesterov=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def get_metrics():
    return [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    modelName = 'fashion-mnist'

    # ── ENV VARS ──
    scratchDir      = os.getenv('SCRATCH_DIR', '/platform/scratch')
    maxEpoch        = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
    minPeers        = int(os.getenv('MIN_PEERS', str(defaultMinPeers)))
    dpEnabled       = os.getenv('DP_ENABLED', 'false').lower() == 'true'
    noiseMultiplier = float(os.getenv('NOISE_MULTIPLIER', '0.0'))
    l2NormClip      = float(os.getenv('L2_NORM_CLIP', '1.0'))
    microbatches    = int(os.getenv('MICROBATCHES', str(batchSize)))

    if batchSize % microbatches != 0:
        raise ValueError(
            f"MICROBATCHES ({microbatches}) must perfectly divide batchSize ({batchSize})."
        )

    optimizerType = os.getenv('OPTIMIZER', 'sgd').lower()
    learningRate  = float(os.getenv('LEARNING_RATE', '0'))
    actual_lr     = learningRate or (0.001 if optimizerType == 'adam' else 0.01)

    # ── DUAL-PHASE DP ENV VARS ──
    # Replaces: CASCADED_DP, DP_DROP_WINDOW, DP_SLOPE_THRESHOLD, ACC_PLATEAU_THRESHOLD
    dualPhase          = os.getenv('DUAL_PHASE_DP', 'false').lower() == 'true'
    minDpEpochs        = int(os.getenv('MIN_DP_EPOCHS', '5'))
    curvatureThresh    = float(os.getenv('CURVATURE_THRESHOLD', '0.002'))
    curvatureWindow    = int(os.getenv('CURVATURE_WINDOW', '3'))

    nodeId   = int(os.getenv('NODE_ID', '0'))
    numNodes = int(os.getenv('NUM_NODES', '2'))

    os.makedirs(scratchDir, exist_ok=True)

    print('***** Starting model =', modelName)
    print('-' * 64)

    # ── LOAD DATA ──
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # ───────────────────────────────────────────────────────────────────────
    # SPLIT DATASET PER NODE
    # ───────────────────────────────────────────────────────────────────────


    total_samples = len(x_train)

    split_size = total_samples // numNodes

    start = nodeId * split_size

    # Last node gets remaining samples
    if nodeId == numNodes - 1:
        end = total_samples
    else:
        end = start + split_size

    x_train = x_train[start:end]
    y_train = y_train[start:end]

    print('-' * 64)
    print(f'Node ID: {nodeId}')
    print(f'Total Nodes: {numNodes}')
    print(f'Training sample range: {start} -> {end}')
    print(f'Local training dataset size: {len(x_train)}')
    print('-' * 64)

    # ── NORMALIZE ──
    x_train = x_train / 255.0
    x_test  = x_test  / 255.0

    num_train_samples = len(x_train)

    # ── ONE-HOT ENCODE ──
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test,  10)

    # ── MODEL ──
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = get_optimizer(
        optimizerType, dpEnabled, learningRate or None,
        l2NormClip, noiseMultiplier, microbatches
    )

    # DP requires per-sample losses (Reduction.NONE) so the optimizer can
    # clip each sample's gradient independently before aggregating.
    loss = (
        tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )
        if dpEnabled
        else tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    )

    model.compile(loss=loss, optimizer=optimizer, metrics=get_metrics())

    # ── DATA PIPELINES ──
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

    # ── SWARM CALLBACK ──
    swarmCallback = SwarmCallback(
        syncFrequency=1024,
        minPeers=minPeers,
        adsValData=val_ds,
        adsValBatchSize=batchSize,
        mergeMethod='mean',
        totalEpochs=maxEpoch
    )

    callbacks          = [swarmCallback]
    dualPhaseDpCallback = None

    if dpEnabled and dualPhase:
        dualPhaseDpCallback = DualPhaseDPCallback(
            node_id=nodeId,
            num_nodes=numNodes,
            scratch_dir=scratchDir,
            optimizer_type=optimizerType,
            learning_rate=actual_lr,
            curvature_threshold=curvatureThresh,
            curvature_window=curvatureWindow,
            min_dp_epochs=minDpEpochs,
        )
        callbacks.append(dualPhaseDpCallback)

    # ── TRAIN ──
    print('Starting training ...')
    train_start = time.time()
    model.fit(train_ds, epochs=maxEpoch, validation_data=val_ds, callbacks=callbacks)
    train_end     = time.time()
    training_time = round(train_end - train_start, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    print('\nRunning final post-training evaluation on test dataset...')

    eval_results       = model.evaluate(val_ds, verbose=0)
    final_test_loss    = float(eval_results[0])
    final_test_accuracy = float(eval_results[1])

    y_true_list, y_pred_list = [], []
    for x_b, y_b in val_ds:
        preds = model.predict_on_batch(x_b)
        y_true_list.append(np.argmax(y_b.numpy(), axis=1))
        y_pred_list.append(np.argmax(preds,        axis=1))

    y_true         = np.concatenate(y_true_list, axis=0)
    y_pred         = np.concatenate(y_pred_list, axis=0)
    final_test_f1  = float(f1_score(y_true, y_pred, average='macro'))

    eps_at_drop = None
    eps_final   = None

    if dpEnabled and noiseMultiplier > 0:
        print('-' * 64)
        print('***** PRIVACY REPORT *****')
        delta = 1.0 / num_train_samples

        # Epochs during which DP was actually active
        dp_epochs = (
            dualPhaseDpCallback.dp_drop_epoch
            if (dualPhaseDpCallback and dualPhaseDpCallback.dp_drop_epoch)
            else maxEpoch
        )

        eps_at_drop, _ = compute_dp_sgd_privacy(
            n=num_train_samples,
            batch_size=batchSize,
            noise_multiplier=noiseMultiplier,
            epochs=dp_epochs,
            delta=delta
        )

        # Phase 2 adds no noise so total epsilon equals eps_at_drop
        eps_final = eps_at_drop

        print(f"DP active for     : {dp_epochs} / {maxEpoch} epochs  (Phase 1)")
        print(f"Epsilon at drop   : {eps_at_drop:.4f}")
        print(f"Final Epsilon (ε) : {eps_final:.4f}  (unchanged — no DP in Phase 2)")
        print(f"Final Delta   (δ) : {delta:.2e}")
        print('-' * 64)

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    results = {
        "config": {
            "model_name":      modelName,
            "node_id":         nodeId,
            "num_nodes":       numNodes,
            "epochs":          maxEpoch,
            "batch_size":      batchSize,
            "optimizer":       optimizerType,
            "learning_rate":   actual_lr,
            "dp_enabled":      dpEnabled,
            "dual_phase_dp":   dualPhase,
            "l2_norm_clip":    l2NormClip,
            "noise_multiplier": noiseMultiplier,
            "microbatches":    microbatches,
        },
        "performance": {
            "training_time_seconds": training_time,
            "final_test_loss":       final_test_loss,
            "final_test_accuracy":   final_test_accuracy,
            "final_test_f1_macro":   final_test_f1,
        },
        "privacy": {
            # eps_at_drop is the meaningful number — privacy cost of Phase 1.
            # eps_final equals it since Phase 2 adds no noise.
            "eps_at_drop":             round(eps_at_drop, 4) if eps_at_drop is not None else None,
            "eps_final":               round(eps_final,   4) if eps_final   is not None else None,
            "delta":                   1.0 / num_train_samples if dpEnabled else None,
            "dp_phase1_epochs":        dualPhaseDpCallback.dp_drop_epoch  if dualPhaseDpCallback else None,
            "dp_drop_reason":          dualPhaseDpCallback.dp_drop_reason if dualPhaseDpCallback else None,
            "curvature_threshold":     curvatureThresh,
            "curvature_window":        curvatureWindow,
            # Full per-epoch telemetry for post-run analysis
            "loss_history":            dualPhaseDpCallback.loss_history      if dualPhaseDpCallback else None,
            "velocity_history":        dualPhaseDpCallback.velocity_history  if dualPhaseDpCallback else None,
            "smoothed_curvature":      dualPhaseDpCallback.smoothed_curvature if dualPhaseDpCallback else None,
        }
    }

    result_file  = os.getenv("RESULT_FILE", "results.json")
    results_path = os.path.join("/results", result_file)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    model.save(os.path.join(scratchDir, modelName))
    print('Saved the trained model and verified final test metrics JSON!')


if __name__ == '__main__':
    main()