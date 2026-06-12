# Fashion-MNIST with Swarm Learning + Differential Privacy

## Overview

This project demonstrates distributed Fashion-MNIST training using:

- HPE Swarm Learning
- TensorFlow/Keras
- Differential Privacy (DP-SGD / DP-Adam)
- TensorFlow Privacy

The implementation supports:

- Standard SGD / Adam training
- Differentially Private SGD and Adam
- Configurable Gaussian noise
- Privacy accounting (Epsilon & Delta reporting)
- Accuracy + F1 Score evaluation
- Automatic JSON result saving
- Automatic ML log collection

---

# Project Structure

```text
fashion-mnist/
├── cert/
├── ml-context/
├── model/
│   └── fashion-mnist.py
├── results/
│   ├── *.json
│   └── *.log
├── tmp/
│   ├── sl1/
│   └── sl2/
└── README.md
```

---

# 0. Clone Project Repository into Workspace

```bash
cd ~/swarm-learning/workspace/

git clone https://github.com/clearlynew/Fashion-MNIST-using-DP.git fashion-mnist
```

---

# 1. Generate Certificates

```bash
cd ~/swarm-learning/

cp -r examples/utils/gen-cert workspace/fashion-mnist/

./workspace/fashion-mnist/gen-cert -e fashion-mnist -i 1

./workspace/fashion-mnist/gen-cert -e fashion-mnist -i 2
```

---

# 2. Delete SWOP/SWCI Certificates

```bash
cd workspace/fashion-mnist/cert

rm swop-* swci-*

cd ../../../
```

---

# 3. Create Docker Network

```bash
docker network create host-1-net
```

---

# 4. Create Required Directories

```bash
mkdir -p ~/swarm-learning/workspace/fashion-mnist/tmp/sl1

mkdir -p ~/swarm-learning/workspace/fashion-mnist/tmp/sl2

mkdir -p ~/swarm-learning/workspace/fashion-mnist/results

chmod -R 777 ~/swarm-learning/workspace/fashion-mnist/tmp

chmod -R 777 ~/swarm-learning/workspace/fashion-mnist/results
```

---

# 5. Copy SwarmLearning Wheel

```bash
cp ~/swarm-learning/lib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl \
~/swarm-learning/workspace/fashion-mnist/ml-context/swarmlearning-0.0.1-py3-none-manylinux_2_24_x86_64.whl
```

---

# 6. Build ML Docker Image

```bash
docker build -t fashion-ml-env \
~/swarm-learning/workspace/fashion-mnist/ml-context
```

---

# 7. Run APLS

```bash
docker run -d \
--name apls \
--network host-1-net \
-v apls-volume:/hpe \
-p 5814:5814 \
--restart unless-stopped \
hub.myenterpriselicense.hpe.com/hpe_eval/autopass/apls:9.19
```

---

# 8. Set Environment Variables

```bash
export HOST_IP=172.1.1.1

export SN_IP=172.1.1.1

export APLS_IP=172.1.1.1

export SN_API_PORT=30304
```

---

# 9. Run SN (Swarm Network Node)

```bash
cd ~/swarm-learning

./scripts/bin/run-sn -d --name=sn1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sentinel \
--sn-api-port=${SN_API_PORT} \
--key=workspace/fashion-mnist/cert/sn-1-key.pem \
--cert=workspace/fashion-mnist/cert/sn-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--apls-ip=${APLS_IP}
```

---

# 10. Monitor SN Until Ready

```bash
docker logs -f sn1
```

Wait until:

```text
swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304
```

---

# Between Experiments

Stop old containers before every new experiment:

```bash
docker rm -f sn1 sl1 sl2 ml1 ml2 2>/dev/null
```

---

# Experiment Summary

| Experiment | DP | Noise | Optimizer |
|---|---|---|---|
| Baseline Adam | No | 0.0 | Adam |
| Baseline SGD | No | 0.0 | SGD |
| DP Adam 0.5 | Yes | 0.5 | Adam |
| DP Adam 1 | Yes | 1.0 | Adam |
| DP Adam 3 | Yes | 3.0 | Adam |
| DP SGD 0.5 | Yes | 0.5 | SGD |
| DP SGD 1 | Yes | 1.0 | SGD |
| DP SGD 3 | Yes | 3.0 | SGD |

---

# Example Experiment — Baseline Adam

## Run SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fashion-mnist/cert/sl-1-key.pem \
--cert=workspace/fashion-mnist/cert/sl-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl1:/tmp/hpe-swarm \
--ml-v workspace/fashion-mnist/model:/tmp/test/model \
--ml-v workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_baseline_adam_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e NODE_ID=0 \
--ml-e NUM_NODES=2 \
--ml-e DP_ENABLED=false \
--ml-e OPTIMIZER=adam \
--ml-e METRIC=both \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml1 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_baseline_adam_ml1.log 2>&1 &
```

---

## Run SL2

```bash
./scripts/bin/run-sl -d --name=sl2 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=17000 \
--key=workspace/fashion-mnist/cert/sl-2-key.pem \
--cert=workspace/fashion-mnist/cert/sl-2-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl2:/tmp/hpe-swarm \
--ml-v workspace/fashion-mnist/model:/tmp/test/model \
--ml-v workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_baseline_adam_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e NODE_ID=1 \
--ml-e NUM_NODES=2 \
--ml-e DP_ENABLED=false \
--ml-e OPTIMIZER=adam \
--ml-e METRIC=both \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml2 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_baseline_adam_ml2.log 2>&1 &
```

---

# Example Experiment — DP Adam (Noise = 1)

## Run SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fashion-mnist/cert/sl-1-key.pem \
--cert=workspace/fashion-mnist/cert/sl-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl1:/tmp/hpe-swarm \
--ml-v workspace/fashion-mnist/model:/tmp/test/model \
--ml-v workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_dp1_adam_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e NODE_ID=0 \
--ml-e NUM_NODES=2 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=1 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e OPTIMIZER=adam \
--ml-e METRIC=both \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml1 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_dp1_adam_ml1.log 2>&1 &
```

---

## Run SL2

```bash
./scripts/bin/run-sl -d --name=sl2 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=17000 \
--key=workspace/fashion-mnist/cert/sl-2-key.pem \
--cert=workspace/fashion-mnist/cert/sl-2-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl2:/tmp/hpe-swarm \
--ml-v workspace/fashion-mnist/model:/tmp/test/model \
--ml-v workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_dp1_adam_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e NODE_ID=1 \
--ml-e NUM_NODES=2 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=1 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e OPTIMIZER=adam \
--ml-e METRIC=both \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml2 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_dp1_adam_ml2.log 2>&1 &
```

---

# Results Storage

All JSON metrics + logs are automatically saved inside:

```text
~/swarm-learning/workspace/fashion-mnist/results/
```

---

# Differential Privacy Parameters

| Parameter | Description |
|---|---|
| `DP_ENABLED` | Enables/disables DP |
| `NOISE_MULTIPLIER` | Gaussian noise multiplier |
| `L2_NORM_CLIP` | Gradient clipping threshold |
| `MICROBATCHES` | Number of microbatches |
| `MAX_EPOCHS` | Training epochs |
| `OPTIMIZER` | SGD or Adam |
| `RESULT_FILE` | Output JSON filename |
| `CASCADED_DP` | Enable dual-plateau-gated DP drop (SNR + accuracy) |
| `SNR_PLATEAU_EPS` | Drop DP when epoch-over-epoch relative SNR change < this (default `0.02`) |
| `ACC_PLATEAU_EPS` | Drop DP when epoch-over-epoch relative accuracy change < this (default `0.005`) |
| `DP_DROP_WINDOW` | Rolling window size for SNR/accuracy averaging |
| `MIN_DP_EPOCHS` | Minimum epochs before DP drop is evaluated |

---

# Expected Results

| Privacy ↑ | Utility ↓ |
|---|---|
| Stronger privacy | Lower accuracy |
| More noise | Slower convergence |
| Higher DP | Longer training time |

---

# Example Experiment — Adaptive DP Adam (Auto-Tuned)

## Run SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fashion-mnist/cert/sl-1-key.pem \
--cert=workspace/fashion-mnist/cert/sl-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_v2.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl1:/tmp/hpe-swarm \
--ml-v workspace/fashion-mnist/model:/tmp/test/model \
--ml-v workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_autotune_adam_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e NODE_ID=0 \
--ml-e NUM_NODES=2 \
--ml-e DP_ENABLED=true \
--ml-e DP_AUTO_TUNE=true \
--ml-e DP_TARGET_EPSILON=10.0 \
--ml-e DP_ADAPTIVE_CALLBACK=true \
--ml-e OPTIMIZER=adam \
--ml-e METRIC=both \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml1 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_autotune_adam_ml1.log 2>&1 &
```

---

## Run SL2

```bash
./scripts/bin/run-sl -d --name=sl2 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=17000 \
--key=workspace/fashion-mnist/cert/sl-2-key.pem \
--cert=workspace/fashion-mnist/cert/sl-2-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_v2.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl2:/tmp/hpe-swarm \
--ml-v workspace/fashion-mnist/model:/tmp/test/model \
--ml-v workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_autotune_adam_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e NODE_ID=1 \
--ml-e NUM_NODES=2 \
--ml-e DP_ENABLED=true \
--ml-e DP_AUTO_TUNE=true \
--ml-e DP_TARGET_EPSILON=10.0 \
--ml-e DP_ADAPTIVE_CALLBACK=true \
--ml-e OPTIMIZER=adam \
--ml-e METRIC=both \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml2 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_autotune_adam_ml2.log 2>&1 &
```

---

## Additional Parameters Used

| Parameter            | Value               |
| -------------------- | ------------------- |
| DP_ENABLED           | true                |
| DP_AUTO_TUNE         | true                |
| DP_TARGET_EPSILON    | 10.0                |
| DP_ADAPTIVE_CALLBACK | true                |
| OPTIMIZER            | adam                |
| MAX_EPOCHS           | 8                   |
| MODEL FILE           | fashion-mnist_v2.py |

### Description

* `DP_AUTO_TUNE=true` automatically profiles the dataset and selects suitable DP parameters.
* `DP_TARGET_EPSILON=10.0` specifies the target privacy budget.
* `DP_ADAPTIVE_CALLBACK=true` enables dynamic adjustment of privacy parameters during training.
* `fashion-mnist_v2.py` includes the adaptive privacy callback and auto-tuning functionality. 

---

# Example Experiment — Cascaded DP Adam (Privacy Cutoff)

Before running the SWARM nodes we need to create a shared directory inside the temp directory

```bash
mkdir -p ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch
chmod 777 ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch
```

## Run SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fashion-mnist/cert/sl-1-key.pem \
--cert=workspace/fashion-mnist/cert/sl-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_v3.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl1:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_cascaded_dp_50eps_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=0 \
--ml-e NUM_NODES=2 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.5 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e CASCADED_DP=true \
--ml-e DP_DROP_WINDOW=5 \
--ml-e MIN_DP_EPOCHS=5 \
--ml-e DP_SLOPE_THRESHOLD=0.04 \
--ml-e ACC_PLATEAU_THRESHOLD=0.001 \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml1 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_cascaded_dp_50eps_ml1.log 2>&1 &
```

---

## Run SL2

```bash
./scripts/bin/run-sl -d --name=sl2 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=17000 \
--key=workspace/fashion-mnist/cert/sl-2-key.pem \
--cert=workspace/fashion-mnist/cert/sl-2-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_v3.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl2:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_cascaded_dp_50eps_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=1 \
--ml-e NUM_NODES=2 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.5 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e CASCADED_DP=true \
--ml-e DP_DROP_WINDOW=5 \
--ml-e MIN_DP_EPOCHS=5 \
--ml-e DP_SLOPE_THRESHOLD=0.04 \
--ml-e ACC_PLATEAU_THRESHOLD=0.001 \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml2 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_cascaded_dp_50eps_ml2.log 2>&1 &
```

---

## Additional Parameters Used

| Parameter | Value |
|------------|--------|
| CASCADED_DP | true |
| DP_DROP_WINDOW | 5 |
| MIN_DP_EPOCHS | 5 |
| DP_SLOPE_THRESHOLD | 0.04 |
| ACC_PLATEAU_THRESHOLD | 0.001 |

### Description

* Cascaded DP begins training with Differential Privacy enabled using DP-Adam.
* The model continuously monitors validation accuracy improvements during training.
* After a minimum warm-up period (`MIN_DP_EPOCHS`), the system computes the accuracy improvement trend.
* If the improvement remains below `DP_SLOPE_THRESHOLD` for `DP_DROP_WINDOW` consecutive epochs, the model is considered to have reached a performance plateau.
* Once a plateau is detected, Differential Privacy is automatically disabled for the remaining epochs.
* This allows privacy protection during the most information-rich phase of learning while avoiding unnecessary utility loss after convergence.
* All swarm nodes coordinate the cutoff decision through a shared scratch directory (`SCRATCH_DIR`) to ensure synchronized DP disabling.
* The approach aims to improve final accuracy compared with always-on DP while still providing privacy protection during the critical early stages of training.

---

## New Features

### Non-IID Data Partitioning

The enhanced implementation supports multiple data partitioning strategies via:

```bash
--ml-e PARTITION_MODE=<mode>
```

Available modes:

| Mode             | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| `iid`            | Uniform distribution across all nodes                       |
| `noniid_equal`   | Label-skewed distribution with equal sample counts per node |
| `noniid_unequal` | Unequal sample counts per node with weighted aggregation    |

---

### Label-Skew Non-IID

Enable using:

```bash
--ml-e PARTITION_MODE=noniid_equal
```

For a 2-node configuration:

```text
Node 0: 80% classes 0-4, 20% classes 5-9
Node 1: 20% classes 0-4, 80% classes 5-9
```

Both nodes contain the same number of samples.

---

### Unequal-Size Non-IID

Enable using:

```bash
--ml-e PARTITION_MODE=noniid_unequal
```

Example (2 nodes):

```text
Node 0: 48,000 samples
Node 1: 12,000 samples
```

Weighted aggregation is automatically enabled:

```text
Node 0 weight = 80
Node 1 weight = 20
```

### Running the Enhanced Version

Use:

```bash
--ml-cmd=/tmp/test/model/fashion-mnist_nonuniform.py
```


# Example Experiment — SNR-Gated Cascaded DP Adam

Before running the SWARM nodes we need to create a shared directory inside the temp directory:

```bash
mkdir -p ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch
chmod 777 ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch
```

## Run SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fashion-mnist/cert/sl-1-key.pem \
--cert=workspace/fashion-mnist/cert/sl-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_snr.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl1:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_snr_cascaded_dp_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=0 \
--ml-e NUM_NODES=2 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.5 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e CASCADED_DP=true \
--ml-e SNR_PLATEAU_EPS=0.02 \
--ml-e ACC_PLATEAU_EPS=0.005 \
--ml-e DP_DROP_WINDOW=5 \
--ml-e MIN_DP_EPOCHS=5 \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml1 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_snr_cascaded_dp_ml1.log 2>&1 &
```

---

## Run SL2

```bash
./scripts/bin/run-sl -d --name=sl2 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=17000 \
--key=workspace/fashion-mnist/cert/sl-2-key.pem \
--cert=workspace/fashion-mnist/cert/sl-2-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_snr.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl2:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_snr_cascaded_dp_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=1 \
--ml-e NUM_NODES=2 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.5 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e CASCADED_DP=true \
--ml-e SNR_PLATEAU_EPS=0.02 \
--ml-e ACC_PLATEAU_EPS=0.005 \
--ml-e DP_DROP_WINDOW=5 \
--ml-e MIN_DP_EPOCHS=5 \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml2 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_snr_cascaded_dp_ml2.log 2>&1 &
```

---

## Additional Parameters Used

| Parameter        | Value                    |
|------------------|--------------------------|
| CASCADED_DP      | true                     |
| SNR_PLATEAU_EPS  | 0.02                     |
| ACC_PLATEAU_EPS  | 0.005                   |
| DP_DROP_WINDOW   | 5                        |
| MIN_DP_EPOCHS    | 5                        |
| NOISE_MULTIPLIER | 0.5                      |
| L2_NORM_CLIP     | 1.0                      |
| MODEL FILE       | fashion-mnist_snr.py     |

### Description

* SNR + Accuracy Dual-Plateau Cascaded DP begins training with Differential Privacy enabled using DP-Adam.
* Each epoch, the callback computes two signals:
  * **Signal-to-Noise Ratio (SNR):** `SNR = grad_norm / noise_std`, where `noise_std = (L2_NORM_CLIP × NOISE_MULTIPLIER) / √batch_size`.
  * **Validation accuracy** from the Keras epoch logs.
* After the minimum warm-up period (`MIN_DP_EPOCHS`), both **relative change rates** are evaluated:
  * `Δ_rel(SNR) = |SNR_now - SNR_prev| / SNR_prev`
  * `Δ_rel(acc) = |acc_now - acc_prev| / acc_prev`
* A node casts a drop-DP vote **only when BOTH conditions are satisfied simultaneously**:
  * `Δ_rel(SNR) < SNR_PLATEAU_EPS` (default `0.02`) — gradient dynamics have plateaued.
  * `Δ_rel(acc) < ACC_PLATEAU_EPS` (default `0.005`) — validation accuracy has plateaued.
* Both triggers are **dimensionless and self-calibrating**: they do not depend on absolute values, so they generalise across architectures and hyperparameter sets.
* A diagnostic log line is printed each epoch showing the status of each condition (`✓ PLATEAU` or `…`), and if a vote is not cast the blocking condition is reported explicitly.
* Once **all nodes** have voted (strict quorum), every node simultaneously drops the DP optimizer and recompiles with a standard Adam optimizer, allowing clean fine-tuning for the remaining epochs.
* The privacy budget (ε) is computed only for the epochs during which DP was actually active.

---
# Experiment — Dual-Phase Loss Curvature DP (Adam)

Before running the SWARM nodes we need to create a shared directory inside the temp directory:

```bash
mkdir -p ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch
chmod 777 ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch
```

## Run SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fashion-mnist/cert/sl-1-key.pem \
--cert=workspace/fashion-mnist/cert/sl-1-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_dualphase.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl1:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_dualphase_dp_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=0 \
--ml-e NUM_NODES=2 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.8 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e DUAL_PHASE_DP=true \
--ml-e CURVATURE_THRESHOLD=0.02 \
--ml-e CURVATURE_WINDOW=7 \
--ml-e MIN_DP_EPOCHS=5 \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml1 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_dualphase_dp_ml1.log 2>&1 &
```

---

## Run SL2

```bash
./scripts/bin/run-sl -d --name=sl2 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=17000 \
--key=workspace/fashion-mnist/cert/sl-2-key.pem \
--cert=workspace/fashion-mnist/cert/sl-2-cert.pem \
--capath=workspace/fashion-mnist/cert/ca/capath \
--ml-image=fashion-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fashion-mnist_dualphase.py \
-v ~/swarm-learning/workspace/fashion-mnist/tmp/sl2:/tmp/hpe-swarm \
--ml-v ~/swarm-learning/workspace/fashion-mnist/tmp/shared_scratch:/tmp/scratch \
--ml-v ~/swarm-learning/workspace/fashion-mnist/model:/tmp/test/model \
--ml-v ~/swarm-learning/workspace/fashion-mnist/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp_dualphase_dp_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=50 \
--ml-e NODE_ID=1 \
--ml-e NUM_NODES=2 \
--ml-e OPTIMIZER=adam \
--ml-e LEARNING_RATE=0.001 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.8 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--ml-e DUAL_PHASE_DP=true \
--ml-e CURVATURE_THRESHOLD=0.02 \
--ml-e CURVATURE_WINDOW=7 \
--ml-e MIN_DP_EPOCHS=5 \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml2 > \
~/swarm-learning/workspace/fashion-mnist/results/exp_dualphase_dp_ml2.log 2>&1 &
```

---

## Additional Parameters Used

| Parameter           | Value                         |
|---------------------|-------------------------------|
| DUAL_PHASE_DP       | true                          |
| CURVATURE_THRESHOLD | 0.02                          |
| CURVATURE_WINDOW    | 7                             |
| MIN_DP_EPOCHS       | 5                             |
| NOISE_MULTIPLIER    | 0.8                           |
| L2_NORM_CLIP        | 1.0                           |
| MODEL FILE          | fashion-mnist_dualphase_loss.py    |

### Description

* Training starts with DP-Adam enabled and is divided into two phases:

  * **Phase 1 (DP active):** DP protects the model during rapid learning and major parameter updates.
  * **Phase 2 (DP disabled):** Once learning stabilizes, training continues with standard Adam for final refinement.

* At each epoch, the method uses only the training loss already computed by Keras:

  * **Velocity:** change in loss between epochs.
  * **Curvature:** change in velocity.
  * **Smoothed curvature:** rolling average of curvature to reduce noise sensitivity.

* After a minimum DP period, a node votes to disable DP when:

  * `smoothed_curvature < CURVATURE_THRESHOLD`, indicating the loss curve has flattened.

* When all nodes agree (quorum), they simultaneously switch from DP-Adam to Adam and continue training without DP.

* Vote files are written atomically and cleaned at startup to avoid stale or partially written votes.

* Privacy accounting is performed only for the DP-active phase, so:

  * `eps_at_drop` = privacy budget consumed before DP is disabled.
  * `eps_final` = `eps_at_drop`, since no additional privacy loss occurs afterward.

* Because it relies solely on existing loss values, this approach introduces virtually no computational overhead compared to methods requiring extra gradient or validation passes.


# Notes

- TensorFlow version: `2.7.0`
- TensorFlow Privacy version: `0.7.3`
- TensorFlow Probability version: `0.15.0`
