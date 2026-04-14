# 📡 5G O-RAN AI Auto-Scaler

An end-to-end closed-loop automation system for dynamically scaling 5G and O-RAN microservices using Predictive AI (LSTM Deep Learning) and Kubernetes.

---

## 🚀 Project Overview

Traditional Kubernetes auto-scaling (like HPA) relies on *reactive* thresholds (e.g., waiting for CPU utilization to exceed 80%). This often leads to temporary service degradation during sudden traffic spikes in telecommunication networks.

This project introduces a **proactive, AI-driven approach**:

1. **Senses** real-time 5G network traffic metrics (Active Users, Throughput, RSRP).
2. **Predicts** future network congestion using a trained LSTM neural network.
3. **Acts** by directly communicating with the Kubernetes API to scale up/down network resources *before* bottlenecks occur.

---

## 🏗 System Architecture

<img width="701" height="571" alt="Screenshot from 2026-04-14 04-32-50" src="https://github.com/user-attachments/assets/ecec83f4-75e5-490b-8b61-210f9fd60b3c" />

The system is built as a **continuous closed-loop pipeline** with 4 stages:

| Stage | Description |
|-------|-------------|
| **1 — Data Engineering** | Simulates realistic 5G cell tower time-series data (1440 data points / 24 hrs) covering RSRP, Throughput (Mbps), and Active User Count. |
| **2 — AI Forecasting (LSTM)** | TensorFlow/Keras model with 2× LSTM layers (50 units) + Dense output. Uses MinMaxScaler with sequence length=10 over 5 epochs to predict the next user count. |
| **3 — Containerization (Docker)** | Inference engine packaged into a production-ready Docker image using `python:3.9-slim` base with `imagePullPolicy: Never`. |
| **4 — Kubernetes Automation** | AI Pod calls the K8s API via RBAC-controlled ServiceAccount. Scales `5g-traffic-service` to **3 replicas** if users > 70, or down to **1 replica** under normal load. |

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Programming** | Python 3.9 |
| **AI & Machine Learning** | TensorFlow, Keras, Scikit-Learn (MinMaxScaler), Pandas, NumPy |
| **DevOps & Cloud-Native** | Kubernetes (K8s), Docker, Kubernetes Python Client (`kubernetes`), RBAC |
| **Telecom Concepts** | 5G Core, O-RAN, Network Slicing Simulation |

---

## 📂 Project Structure

```
.
├── network_gen.py              # 5G traffic data simulator (generates network_data.csv)
├── traffic_forecaster.py       # LSTM model training + K8s scaling logic
├── requirements.txt            # Python dependencies
├── Dockerfile.dockerfile       # Production Docker image definition
├── k8s-manifests/
│   ├── serviceaccount.yaml     # ServiceAccount + RBAC ClusterRole/Binding
│   ├── deployment.yaml         # AI forecaster Deployment spec
│   └── service.yaml            # 5g-traffic-service definition
└── images/
    └── architecture.png        # System architecture diagram
```

---

## ⚙️ How to Run Locally

### Prerequisites

- Docker installed and running.
- A local Kubernetes cluster (e.g., [Minikube](https://minikube.sigs.k8s.io/) or [kind](https://kind.sigs.k8s.io/)).
- `kubectl` configured to point to your cluster.
- Python 3.9+

> **Windows users:** Ensure Docker Desktop is configured to use the WSL2 backend for optimal Kubernetes performance.

---

### Step 1 — Generate Network Data

Run the simulator to generate the time-series 5G network traffic data. This produces `network_data.csv` with 1440 rows representing 24 hours of simulated cell tower metrics:

```bash
python network_gen.py
```

---

### Step 2 — Build the Docker Image

```bash
docker build -t 5g-traffic-forecaster:latest -f Dockerfile.dockerfile .
```

> **Note:** The image uses `imagePullPolicy: Never` so Kubernetes uses the locally built image without pushing to a registry.

If using **Minikube**, load the image into the cluster:

```bash
minikube image load 5g-traffic-forecaster:latest
```

---

### Step 3 — Deploy to Kubernetes

Apply all manifests (ServiceAccount, RBAC, Deployment, Service):

```bash
kubectl apply -f k8s-manifests/
```

---

### Step 4 — Monitor the AI Forecaster

Stream live logs from the forecaster pod:

```bash
kubectl logs deploy/ai-forecaster-deployment -f
```

---

### 📸 Expected Output & Live Results

**During Normal Traffic:**

```
🧠 AI Model is ready to predict traffic!
🔮 Predicted future user count: 49
✅ Normal traffic (49 users). Scaling down to 1 replica.
🚀 Scaled '5g-traffic-service' to 1 replicas.
```

**During a Traffic Spike:**

```
🔮 Predicted future user count: 85
⚠️ High traffic predicted (85 users)! Scaling up to 3 replicas.
🚀 Scaled '5g-traffic-service' to 3 replicas.
```

**Live terminal output:**

<img width="1915" height="689" alt="Screenshot from 2026-04-14 04-09-42" src="https://github.com/user-attachments/assets/6c2a6958-b477-4df9-8372-15bf94e5c4a9" />

---

## 📊 Model Performance

Training results observed during a 5-epoch run:

| Epoch | Loss |
|-------|------|
| 1/5 | 0.0742 |
| 2/5 | 0.0039 |
| 3/5 | 0.0034 |
| 4/5 | 0.0036 |
| 5/5 | 0.0036 |

The model converges quickly to low loss values, demonstrating effective learning of the traffic patterns from the simulated dataset.

> **Note:** GPU is not required. The system runs on CPU only. CUDA warnings at startup (if any) are informational and do not affect functionality.

---

## 🔐 Kubernetes RBAC

The AI pod is granted minimal permissions via a dedicated `ServiceAccount` and `ClusterRole`:

```yaml
rules:
  - apiGroups: ["apps"]
    resources: ["deployments/scale"]
    verbs: ["get", "patch"]
```

This follows the **principle of least privilege** — the pod can only read and patch deployment scales, nothing else.

---

## 🔄 Closed-Loop Automation Flow

```
[Network Data] ──► [LSTM Prediction] ──► [Decision Logic]
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        │                                             │
                  users > 70                                   users ≤ 70
                        │                                             │
              Scale UP → 3 replicas                      Scale DOWN → 1 replica
                        │                                             │
                        └──────────────── [K8s API Patch] ───────────┘
                                                │
                                        [Loop continues...]
```

---

## 🚧 Future Improvements

- [ ] Integrate with real O-RAN RIC (RAN Intelligent Controller) via A1/E2 interfaces
- [ ] Replace simulated data with live Prometheus/OpenTelemetry metrics
- [ ] Add Horizontal Pod Autoscaler (HPA) as fallback alongside AI predictions
- [ ] Implement model retraining pipeline with MLflow tracking
- [ ] Add Grafana dashboard for real-time visualization of predictions vs. actual traffic
- [ ] Support multi-slice network scaling (eMBB, URLLC, mMTC)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ at the intersection of AI, 5G, and Cloud-Native infrastructure.*
