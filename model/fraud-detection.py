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
import csv
import tensorflow as tf
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

from swarmlearning.tf import SwarmCallback

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE PARAMETERS (set via --ml-e flags in run-sl command)
# ─────────────────────────────────────────────────────────────────────────────
#
#  DP_ENABLED        Enable Differential Privacy           true | false            (default: false)
#  NOISE_MULTIPLIER  Gaussian noise multiplier             float                   (default: 0.0)
#  L2_NORM_CLIP      Gradient clipping threshold           float                   (default: 1.0)
#  MICROBATCHES      Number of microbatches for DP         int                     (default: = batchSize)
#
#  OPTIMIZER         Optimizer to use                      sgd | adam              (default: sgd)
#  LEARNING_RATE     Learning rate override                float                   (default: 0.01 sgd / 0.001 adam)
#
#  METRIC            Which AUC to report                   auc_roc | auc_pr | both (default: auc_roc)
#
#  MAX_EPOCHS        Training epochs                       int                     (default: 100)
#  MIN_PEERS         Min swarm peers before sync           int                     (default: 2)
#  DATA_DIR          Path to data directory                                        (default: /platform/data)
#  SCRATCH_DIR       Path to scratch/output directory                              (default: /platform/scratch)
#
# ─────────────────────────────────────────────────────────────────────────────

testFileName  = 'SB19_CCFDUBL_TEST.csv'
trainFileName = 'SB19_CCFDUBL_TRAIN.csv'

batchSize       = 32
defaultMaxEpoch = 100
defaultMinPeers = 2


def getXY(dataSet):
    np.random.shuffle(dataSet)
    length = np.size(dataSet, 0)
    X = dataSet[0:length, :-1]
    y = dataSet[0:length, -1:]
    return X, y


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
        else:  # default: sgd
            from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
            print("***** Using DP-SGD optimizer")
            return DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=learning_rate or 0.01
            )

    else:  # non-DP
        if optimizer_type == 'adam':
            print("***** Using Adam optimizer")
            return tf.keras.optimizers.Adam(learning_rate=learning_rate or 0.001)
        else:  # default: sgd
            print("***** Using SGD optimizer")
            return tf.keras.optimizers.SGD(
                learning_rate=learning_rate or 0.01,
                decay=1e-6,
                momentum=0.9,
                nesterov=True
            )


def get_metrics(metric_type):
    """Return Keras metric list."""
    if metric_type == 'auc_pr':
        return [tf.keras.metrics.AUC(curve='PR', name='auc_pr')]
    elif metric_type == 'both':
        return [
            tf.keras.metrics.AUC(curve='ROC', name='auc_roc'),
            tf.keras.metrics.AUC(curve='PR',  name='auc_pr')
        ]
    else:  # default: auc_roc
        return [tf.keras.metrics.AUC(curve='ROC', name='auc_roc')]


def main():
    modelName = 'fraud-detection'

    # ── Read env vars ──────────────────────────────────────────────────────
    dataDir        = os.getenv('DATA_DIR',    '/platform/data')
    scratchDir     = os.getenv('SCRATCH_DIR', '/platform/scratch')
    maxEpoch       = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
    minPeers       = int(os.getenv('MIN_PEERS',  str(defaultMinPeers)))

    dpEnabled       = os.getenv('DP_ENABLED',       'false').lower() == 'true'
    noiseMultiplier = float(os.getenv('NOISE_MULTIPLIER', '0.0'))
    l2NormClip      = float(os.getenv('L2_NORM_CLIP',     '1.0'))
    microbatches    = int(os.getenv('MICROBATCHES', str(batchSize)))

    optimizerType  = os.getenv('OPTIMIZER',     'sgd').lower()
    learningRate   = float(os.getenv('LEARNING_RATE', '0'))  # 0 = use per-optimizer default

    metricType     = os.getenv('METRIC', 'auc_roc').lower()

    os.makedirs(scratchDir, exist_ok=True)
    print('***** Starting model =', modelName)
    print('-' * 64)

    # ── Load data ──────────────────────────────────────────────────────────
    trainFile = dataDir + '/' + trainFileName
    print(f"Loading train dataset {trainFile} ..")
    with open(trainFile, 'r') as f:
        trainData = np.array(list(csv.reader(f, delimiter=","))[1:], dtype=float)
        num_train_samples = np.size(trainData, 0)
        print(f"Size of training dataset: {num_train_samples}")

    print('-' * 64)
    testFile = dataDir + '/' + testFileName
    print(f"Loading test dataset {testFile} ..")
    with open(testFile, 'r') as f:
        testData = np.array(list(csv.reader(f, delimiter=","))[1:], dtype=float)
        print(f"Size of test dataset: {np.size(testData, 0)}")

    print('-' * 64)

    x_train, y_train = getXY(trainData)
    x_test,  y_test  = getXY(testData)

    # ── Model: logistic regression ─────────────────────────────────────────
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        1, input_shape=(30,), activation='sigmoid',
        kernel_initializer='random_uniform', bias_initializer='zeros'
    ))

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = get_optimizer(
        optimizerType, dpEnabled, learningRate or None,
        l2NormClip, noiseMultiplier, microbatches
    )

    # ── Loss (Reduction.NONE required for DP compatibility) ────────────────
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE
    )

    # ── Metrics ────────────────────────────────────────────────────────────
    metrics = get_metrics(metricType)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(model.summary())

    # ── Data pipelines ─────────────────────────────────────────────────────
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(len(x_train)).batch(batchSize, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchSize)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # ── Swarm callback ─────────────────────────────────────────────────────
    swarmCallback = SwarmCallback(
        syncFrequency=128,
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

    # ── Save results JSON ──────────────────────────────────────────────────
    results = {
        "config": {
            "dp_enabled":       dpEnabled,
            "noise_multiplier": noiseMultiplier,
            "l2_norm_clip":     l2NormClip,
            "microbatches":     microbatches,
            "optimizer":        optimizerType,
            "learning_rate":    learningRate or "default",
            "metric":           metricType,
            "epochs":           maxEpoch,
        },
        "privacy": {
            "epsilon": round(eps, 4) if eps is not None else None,
            "delta":   float(1.0 / num_train_samples) if dpEnabled else None,
        },
        "results": {
            name: round(float(val), 4)
            for name, val in zip(score_names, scores)
        },
        "timing": {
            "training_time_seconds": training_time,
            "training_time_minutes": round(training_time / 60, 2)
        }
    }
    
    result_file = os.getenv("RESULT_FILE", "results.json")
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
