############################################################################
## model.py  —  Fashion MNIST Swarm Learning with Adaptive DP
## Drop-in replacement for your original model.py.
##
## New env vars:
##   DP_AUTO_TUNE=true          → enable auto-tuner (overrides NOISE_MULTIPLIER
##                                and L2_NORM_CLIP if set)
##   DP_TARGET_EPSILON=10.0     → privacy budget the tuner aims for
##   DP_ADAPTIVE_CALLBACK=true  → enable self-healing mid-training adjustments
############################################################################

import os
import json
import time
import numpy as np
import tensorflow as tf
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

    if adaptive_cb is not None:
        callbacks.append(adaptive_cb)   # ← self-healing DP callback

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

        eps, _ = compute_dp_sgd_privacy(
            n=num_train_samples,
            batch_size=batchSize,
            noise_multiplier=final_noise,
            epochs=maxEpoch,
            delta=delta,
        )

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
            "auto_tuned":       dpAutoTune,
            "noise_multiplier": final_noise,
            "l2_norm_clip":     final_clip,
            "microbatches":     microbatches,
            "optimizer":        optimizerType,
            "learning_rate":    learningRate or "default",
            "epochs":           maxEpoch,
        },

        "privacy": {
            "epsilon": round(eps, 4) if eps is not None else None,
            "delta":   float(1.0 / num_train_samples) if dpEnabled else None,
        },

        "results": {
            **{name: round(float(val), 4) for name, val in zip(score_names, scores)},
            "f1_score": round(float(f1), 4),
        },

        "timing": {
            "training_time_seconds": training_time,
            "training_time_minutes": round(training_time / 60, 2),
        },

        # ── New sections added by auto-tuner ──────────────────────────
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
