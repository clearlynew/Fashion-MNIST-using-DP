############################################################################
## (C)Copyright 2021-2023 Hewlett Packard Enterprise Development LP
## Licensed under the Apache License, Version 2.0 (the "License"); you may
## not use this file except in compliance with the License. You may obtain
## a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
############################################################################

import os
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

from swarmlearning.tf import SwarmCallback

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE PARAMETERS (set via --ml-e flags in run-sl command)
# ─────────────────────────────────────────────────────────────────────────────
#
#  DP_ENABLED        Enable Differential Privacy           true | false
#  NOISE_MULTIPLIER  Gaussian noise multiplier             float
#  L2_NORM_CLIP      Gradient clipping threshold           float
#  MICROBATCHES      Number of microbatches for DP         int
#
#  OPTIMIZER         Optimizer to use                      sgd | adam
#  LEARNING_RATE     Learning rate override                float
#
#  MAX_EPOCHS        Training epochs                       int
#  MIN_PEERS         Min swarm peers before sync           int
#  SCRATCH_DIR       Path to scratch/output directory
#
# ─────────────────────────────────────────────────────────────────────────────

batchSize       = 32
defaultMaxEpoch = 20
defaultMinPeers = 2


def get_optimizer(optimizer_type, dp_enabled, learning_rate,
                  l2_norm_clip, noise_multiplier, microbatches):
    """Return the correct optimizer (DP or standard) based on config."""

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
                learning_rate=learning_rate or 0.01,
                decay=1e-6,
                momentum=0.9,
                nesterov=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# ONLY METRIC CHANGE
# ─────────────────────────────────────────────────────────────────────────────

def get_metrics():
    """Return Keras metric list for multiclass classification."""
    return [
        tf.keras.metrics.CategoricalAccuracy(name='accuracy')
    ]


def main():
    modelName = 'fashion-mnist'

    # ── Read env vars ──────────────────────────────────────────────────────
    scratchDir     = os.getenv('SCRATCH_DIR', '/platform/scratch')
    maxEpoch       = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
    minPeers       = int(os.getenv('MIN_PEERS',  str(defaultMinPeers)))

    dpEnabled       = os.getenv('DP_ENABLED',       'false').lower() == 'true'
    noiseMultiplier = float(os.getenv('NOISE_MULTIPLIER', '0.0'))
    l2NormClip      = float(os.getenv('L2_NORM_CLIP',     '1.0'))
    microbatches    = int(os.getenv('MICROBATCHES', str(batchSize)))

    optimizerType  = os.getenv('OPTIMIZER',      'sgd').lower()
    learningRate   = float(os.getenv('LEARNING_RATE', '0'))

    os.makedirs(scratchDir, exist_ok=True)

    print('***** Starting model =', modelName)
    print('-' * 64)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading Fashion MNIST dataset ..")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train / 255.0
    x_test  = x_test  / 255.0

    num_train_samples = len(x_train)

    print(f"Size of training dataset: {num_train_samples}")
    print(f"Size of test dataset: {len(x_test)}")
    print('-' * 64)

    # One-hot encode labels for categorical crossentropy
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test,  10)

    # ── Model: ANN ─────────────────────────────────────────────────────────
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = get_optimizer(
        optimizerType,
        dpEnabled,
        learningRate or None,
        l2NormClip,
        noiseMultiplier,
        microbatches
    )

    # ───────────────────────────────────────────────────────────────────────
    # ONLY LOSS CHANGE
    # ───────────────────────────────────────────────────────────────────────

    if dpEnabled:
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False
        )

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = get_metrics()

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    print(model.summary())

    # ── Data pipelines ─────────────────────────────────────────────────────
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    train_ds = train_ds.shuffle(num_train_samples).batch(
        batchSize,
        drop_remainder=True
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).batch(batchSize)

    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # ───────────────────────────────────────────────────────────────────────
    # ONLY SYNC CHANGE
    # ───────────────────────────────────────────────────────────────────────

    swarmCallback = SwarmCallback(
        syncFrequency=1024,
        minPeers=minPeers,
        adsValData=val_ds,
        adsValBatchSize=batchSize,
        mergeMethod='mean',
        totalEpochs=maxEpoch
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print('Starting training ...')

    train_start = time.time()

    model.fit(
        train_ds,
        epochs=maxEpoch,
        validation_data=val_ds,
        callbacks=[swarmCallback]
    )

    train_end     = time.time()
    training_time = round(train_end - train_start, 2)

    print('Training done!')
    print(f"***** Training time: {training_time}s ({round(training_time / 60, 2)} min)")

    # ── Privacy report ─────────────────────────────────────────────────────
    eps = None

    if dpEnabled and noiseMultiplier > 0:
        print('-' * 64)
        print('***** PRIVACY REPORT *****')

        delta = 1.0 / num_train_samples

        eps, _ = compute_dp_sgd_privacy(
            n=num_train_samples,
            batch_size=batchSize,
            noise_multiplier=noiseMultiplier,
            epochs=maxEpoch,
            delta=delta
        )

        print(f"Final Epsilon (ε): {eps:.4f}")
        print(f"Final Delta   (δ): {delta:.2e}")

        print('**************************')
        print('-' * 64)

    elif dpEnabled and noiseMultiplier <= 0:
        print("***** WARNING: noise_multiplier is 0.0 — privacy budget is infinite.")

    # ── Evaluate ───────────────────────────────────────────────────────────
    scores      = model.evaluate(val_ds, verbose=1)
    score_names = ['loss'] + [m.name for m in metrics]

    for name, val in zip(score_names, scores):
        print(f"***** Test {name}: {val:.4f}")

    # ── F1 Score (via sklearn) ─────────────────────────────────────────────
    y_pred         = model.predict(val_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    f1 = f1_score(
        y_true_classes,
        y_pred_classes,
        average='weighted'
    )

    print(f"***** Test f1_score: {f1:.4f}")

    # ── Save results JSON ──────────────────────────────────────────────────
    results = {
        "config": {
            "dp_enabled":       dpEnabled,
            "noise_multiplier": noiseMultiplier,
            "l2_norm_clip":     l2NormClip,
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
            **{
                name: round(float(val), 4)
                for name, val in zip(score_names, scores)
            },
            "f1_score": round(float(f1), 4)
        },

        "timing": {
            "training_time_seconds": training_time,
            "training_time_minutes": round(training_time / 60, 2)
        }
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
