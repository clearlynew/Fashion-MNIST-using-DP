# Fraud Detection with Swarm Learning + Differential Privacy

## Overview

This project demonstrates distributed fraud detection training using:

* HPE Swarm Learning
* TensorFlow/Keras
* Differential Privacy (DP-SGD / DP-Adam)
* TensorFlow Privacy

The implementation supports:

* Standard SGD / Adam training
* Differentially Private SGD and Adam
* Configurable Gaussian noise
* Privacy accounting (Epsilon & Delta reporting)
* AUC-ROC and AUC-PR metrics
* Automatic JSON result saving
* Automatic ML log collection

---

# Project Structure

```text
fraud-detection/
├── cert/
├── data-and-scratch1/
├── data-and-scratch2/
├── ml-context/
├── model/
│   └── fraud-detection.py
├── results/
│   ├── *.json
│   └── *.log
├── tmp/
│   ├── sl1/
│   └── sl2/
└── README.md
```

---

# 1. Clone Project Repository into Workspace

```bash
cd ~/swarm-learning/workspace/

git clone https://github.com/clearlynew/Fraud-Detection-Without-SWOP-SWCI-using-DP.git fraud-detection
```

---

# 2. Generate Certificates

```bash
cd ~/swarm-learning/

cp -r examples/utils/gen-cert workspace/fraud-detection/

./workspace/fraud-detection/gen-cert -e fraud-detection -i 1

./workspace/fraud-detection/gen-cert -e fraud-detection -i 2
```

---

# 3. Delete certificates with "swop" and "swci" in their name

```bash
cd workspace/fraud-detection/cert

rm swop-* swci-*

cd ../../../
```

---

# 4. Create Docker Network

```bash
docker network create host-1-net
```

---

# 5. Create Required Directories

```bash
mkdir -p ~/swarm-learning/workspace/fraud-detection/tmp/sl1

mkdir -p ~/swarm-learning/workspace/fraud-detection/tmp/sl2

mkdir -p ~/swarm-learning/workspace/fraud-detection/results

chmod -R 777 ~/swarm-learning/workspace/fraud-detection/tmp

chmod -R 777 ~/swarm-learning/workspace/fraud-detection/results
```

---

# 6. Copy SwarmLearning Wheel and Remove Duplicate

```bash
cp ~/swarm-learning/lib/swarmlearning-*.whl \
~/swarm-learning/workspace/fraud-detection/ml-context/

rm workspace/fraud-detection/ml-context/swarmlearning-client-*.whl 2>/dev/null
```

---

# 7. Build ML Docker Image

```bash
docker build -t fraud-ml-env ~/swarm-learning/workspace/fraud-detection/ml-context
```

---

# 8. Run APLS

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

# Set Environment Variables

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
--key=workspace/fraud-detection/cert/sn-1-key.pem \
--cert=workspace/fraud-detection/cert/sn-1-cert.pem \
--capath=workspace/fraud-detection/cert/ca/capath \
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

| Exp | DP  | Noise | Epochs | Optimizer |
| --- | --- | ----- | ------ | --------- |
| 1   | No  | 0.0   | 8      | SGD       |
| 2   | Yes | 0.1   | 8      | SGD       |
| 3   | Yes | 0.5   | 8      | SGD       |
| 4   | Yes | 1.0   | 8      | SGD       |
| 5   | Yes | 0.5   | 4      | SGD       |
| 6   | Yes | 0.5   | 8      | Adam      |

---

# Experiment 1 — Baseline

## Run SL1

```bash
./scripts/bin/run-sl -d --name=sl1 \
--network=host-1-net \
--host-ip=${HOST_IP} \
--sn-ip=${SN_IP} \
--sn-api-port=${SN_API_PORT} \
--sl-fs-port=16000 \
--key=workspace/fraud-detection/cert/sl-1-key.pem \
--cert=workspace/fraud-detection/cert/sl-1-cert.pem \
--capath=workspace/fraud-detection/cert/ca/capath \
--ml-image=fraud-ml-env \
--ml-name=ml1 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fraud-detection.py \
-v ~/swarm-learning/workspace/fraud-detection/tmp/sl1:/tmp/hpe-swarm \
--ml-v workspace/fraud-detection/model:/tmp/test/model \
--ml-v workspace/fraud-detection/data-and-scratch1/app-data:/app-data \
--ml-v workspace/fraud-detection/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp1_baseline_sl1.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e DP_ENABLED=false \
--ml-e OPTIMIZER=sgd \
--ml-e METRIC=both \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml1 > ~/swarm-learning/workspace/fraud-detection/results/exp1_baseline_ml1.log 2>&1 &
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
--key=workspace/fraud-detection/cert/sl-2-key.pem \
--cert=workspace/fraud-detection/cert/sl-2-cert.pem \
--capath=workspace/fraud-detection/cert/ca/capath \
--ml-image=fraud-ml-env \
--ml-name=ml2 \
--ml-entrypoint=python3 \
--ml-cmd=/tmp/test/model/fraud-detection.py \
-v ~/swarm-learning/workspace/fraud-detection/tmp/sl2:/tmp/hpe-swarm \
--ml-v workspace/fraud-detection/model:/tmp/test/model \
--ml-v workspace/fraud-detection/data-and-scratch2/app-data:/app-data \
--ml-v workspace/fraud-detection/results:/results \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e RESULT_FILE=exp1_baseline_sl2.json \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e DP_ENABLED=false \
--ml-e OPTIMIZER=sgd \
--ml-e METRIC=both \
--apls-ip=${APLS_IP}
```

Save logs:

```bash
docker logs -f ml2 > ~/swarm-learning/workspace/fraud-detection/results/exp1_baseline_ml2.log 2>&1 &
```

---

# Experiments 2–10

Use the same commands while changing:

* `RESULT_FILE`
* `NOISE_MULTIPLIER`
* `MAX_EPOCHS`
* `OPTIMIZER`

according to the experiment table above.

---

# Results Storage

Each experiment automatically saves:

* JSON metrics/results
* Full ML training logs

inside:

```text
workspace/fraud-detection/results/
```

---

# Differential Privacy Parameters

| Parameter          | Description                 |
| ------------------ | --------------------------- |
| `DP_ENABLED`       | Enables/disables DP         |
| `NOISE_MULTIPLIER` | Gaussian noise multiplier   |
| `L2_NORM_CLIP`     | Gradient clipping threshold |
| `MICROBATCHES`     | Number of microbatches      |
| `MAX_EPOCHS`       | Number of training epochs   |
| `OPTIMIZER`        | SGD or Adam                 |
| `METRIC`           | auc_roc / auc_pr / both     |
| `RESULT_FILE`      | Output JSON filename        |

---

# Expected Results

| Privacy ↑        | Utility ↓            |
| ---------------- | -------------------- |
| Stronger privacy | Lower AUC            |
| More noise       | Slower convergence   |
| Higher DP        | Longer training time |

---

# Notes

* TensorFlow version: `2.7.0`
* TensorFlow Privacy version: `0.7.3`
* TensorFlow Probability version: `0.15.0`
