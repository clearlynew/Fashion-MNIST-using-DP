# Fraud Detection with Swarm Learning + Differential Privacy

## Overview

This project demonstrates distributed fraud detection training using:

- HPE Swarm Learning
- TensorFlow/Keras
- Differential Privacy (DP-SGD)
- TensorFlow Privacy

The implementation supports:

- Standard SGD training
- Differentially Private SGD (DP-SGD)
- Configurable Gaussian noise
- Privacy accounting (Epsilon & Delta reporting)

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

# 4. Create Docker Network (if not already created)

```bash
docker network create host-1-net
```

---

# 5. Create Separate Mount Directory

```bash
mkdir -p ~/swarm-learning/workspace/fraud-detection/tmp/sl1

mkdir -p ~/swarm-learning/workspace/fraud-detection/tmp/sl2

chmod -R 777 ~/swarm-learning/workspace/fraud-detection/tmp
```

---

# 6. Copy SwarmLearning Wheel and delete duplicate

```bash
cp ~/swarm-learning/lib/swarmlearning-*.whl \
~/swarm-learning/workspace/fraud-detection/ml-context/

rm workspace/fraud-detection/ml-context/swarmlearning-client-*.whl 2>/dev/null
```

---

# 7. Update Python Dependencies

Update `ml-context/requirements.txt`:

```text
keras
pandas
protobuf==3.15.6
opencv-python
tensorflow-privacy==0.7.3
tensorflow-probability==0.15.0
```

---

# 8. Build ML Docker Image

```bash
docker build -t fraud-ml-env ~/swarm-learning/workspace/fraud-detection/ml-context
```

---

# 9. Run APLS (only if not running or not connected)

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

# Set Environment Variables (according to hostname -I)

```bash
export HOST_IP=172.1.1.1

export SN_IP=172.1.1.1

export APLS_IP=172.1.1.1

export SN_API_PORT=30304
```

---

# 10. Run SN (Swarm Network Node)

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

# 11. Monitor SN until ready

```bash
docker logs -f sn1
```

Wait until you see:

```text
swarm.blCnt : INFO : Starting SWARM-API-SERVER on port: 30304
```

---

# 12. Baseline Training (WITHOUT Differential Privacy)

## Run SL1 (Baseline)

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
-v ~/workspace/fraud-detection/tmp/sl1:/tmp/hpe-swarm \
--ml-v workspace/fraud-detection/model:/tmp/test/model \
--ml-v workspace/fraud-detection/data-and-scratch1/app-data:/app-data \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e DP_ENABLED=false \
--apls-ip=${APLS_IP}
```

---

## Run SL2 (Baseline)

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
-v ~/workspace/fraud-detection/tmp/sl2:/tmp/hpe-swarm \
--ml-v workspace/fraud-detection/model:/tmp/test/model \
--ml-v workspace/fraud-detection/data-and-scratch2/app-data:/app-data \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e DP_ENABLED=false \
--apls-ip=${APLS_IP}
```

---

# 13. Differential Privacy Training (DP-SGD Enabled)

## Run SL1 (DP Enabled)

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
-v ~/workspace/fraud-detection/tmp/sl1:/tmp/hpe-swarm \
--ml-v workspace/fraud-detection/model:/tmp/test/model \
--ml-v workspace/fraud-detection/data-and-scratch1/app-data:/app-data \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.1 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--apls-ip=${APLS_IP}
```

---

## Run SL2 (DP Enabled)

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
-v ~/workspace/fraud-detection/tmp/sl2:/tmp/hpe-swarm \
--ml-v workspace/fraud-detection/model:/tmp/test/model \
--ml-v workspace/fraud-detection/data-and-scratch2/app-data:/app-data \
--ml-e DATA_DIR=/app-data \
--ml-e SCRATCH_DIR=/tmp/scratch \
--ml-e MIN_PEERS=2 \
--ml-e MAX_EPOCHS=8 \
--ml-e DP_ENABLED=true \
--ml-e NOISE_MULTIPLIER=0.1 \
--ml-e L2_NORM_CLIP=1.0 \
--ml-e MICROBATCHES=32 \
--apls-ip=${APLS_IP}
```

---

# 14. Monitor Training

```bash
docker logs -f sl1

docker logs -f sl2
```

---

# Differential Privacy Parameters

| Parameter          | Description                 |
| ------------------ | --------------------------- |
| `DP_ENABLED`       | Enables/disables DP-SGD     |
| `NOISE_MULTIPLIER` | Gaussian noise multiplier   |
| `L2_NORM_CLIP`     | Gradient clipping threshold |
| `MICROBATCHES`     | Number of microbatches      |
| `MAX_EPOCHS`       | Training epochs             |

---

# Example Experimental Configurations

| Experiment | Noise Multiplier |
| ---------- | ---------------- |
| Baseline   | 0                |
| Weak DP    | 0.1              |
| Medium DP  | 0.5              |
| Strong DP  | 1.0              |

---

# Expected Results

Increasing Differential Privacy strength generally causes:

| Privacy ↑        | Utility ↓            |
| ---------------- | -------------------- |
| Stronger privacy | Lower AUC            |
| More noise       | Slower convergence   |
| Higher DP        | Longer training time |

---

# New Features Added to `fraud-detection.py`

The original training script was modified to support Differential Privacy.

## Added Features

### DP-SGD Optimizer Support

```python
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
```

### Epsilon & Delta Privacy Accounting

```python
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
```

### Dynamic DP Configuration via Environment Variables

```python
DP_ENABLED
NOISE_MULTIPLIER
L2_NORM_CLIP
MICROBATCHES
```

### Per-example Loss Support

```python
reduction=tf.keras.losses.Reduction.NONE
```

### TensorFlow Dataset Pipeline

Added optimized dataset pipelines:

```python
tf.data.Dataset
```

with batching, prefetching, and shuffle support.

### Privacy Report Generation

Training now outputs:

```text
Final Epsilon (ε)
Final Delta (δ)
```

after DP training.

---

# Notes

- TensorFlow version used: `2.7.0`
- TensorFlow Privacy version used: `0.7.3`
- TensorFlow Probability version used: `0.15.0`

---

