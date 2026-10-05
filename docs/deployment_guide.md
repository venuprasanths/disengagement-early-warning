# Production & Cloud Deployment Guide
## Transparent Disengagement Early-Warning System for Schools

This guide documents the procedures for packaging, deploying, and maintaining the Transparent Disengagement Early-Warning System in local sandbox, containerized, and enterprise school district cloud environments.

---

## 1. Deployment Architecture Overview

```
                        +------------------------------------------+
                        |   School District SIS & LMS Data Feeds   |
                        |   (OneRoster v1.2, Canvas, PowerSchool)   |
                        +--------------------+---------------------+
                                             |
                                             v  [Weekly TLS 1.3 Ingestion]
                        +------------------------------------------+
                        |        UnifiedIngestionPipeline          |
                        |     (Pydantic Schema & Leakage Gate)     |
                        +--------------------+---------------------+
                                             |
                     +-----------------------+-----------------------+
                     |                                               |
                     v                                               v
       +----------------------------+                 +----------------------------+
       |   Academic Counselor &     |                 |    Student Self-Advocacy   |
       |   Staff Analytics Portal   |                 |      Reflection Portal     |
       |   - Policy Slider (tau)    |                 |   - Strengths Assets       |
       |   - Local TreeSHAP Cards   |                 |   - Study Pacing Trends    |
       |   - Reliability Diagrams   |                 |   - Private Pulse Check-In |
       |   - FERPA Memo Generator   |                 |   - Office Hours / Tutoring|
       +----------------------------+                 +----------------------------+
                     |                                               |
                     +-----------------------+-----------------------+
                                             |
                                             v
                        +------------------------------------------+
                        |     MLOps Drift & Health Supervisor      |
                        |    (Monthly PSI Audit & Auto-Retrain)    |
                        +------------------------------------------+
```

---

## 2. Local Containerized Deployment (Docker & Docker Compose)

The fastest way to deploy the system locally with all dependencies pinned:

### Prerequisites
- Docker Engine $\ge 24.0$
- Docker Compose $\ge 2.20$

### 2.1 One-Command Launch
From the repository root:
```bash
docker compose up --build -d
```

### 2.2 Verifying Service Health
Check container logs and operational health:
```bash
# View container status
docker compose ps

# Inspect live application logs
docker compose logs -f early-warning-dashboard

# Query internal health probe
curl -f http://localhost:8501/_stcore/health
```

### 2.3 Accessing the Application
- **URL**: `http://localhost:8501`
- **Port**: `8501` (configurable in `docker-compose.yml`)

### 2.4 Teardown
```bash
docker compose down
```

---

## 3. Local Bare-Metal Setup (Without Docker)

### Windows (PowerShell)
```powershell
# 1. Setup virtual environment & dependencies
.\run.ps1 setup

# 2. Run automated test suite (17 tests)
.\run.ps1 test

# 3. Launch interactive Streamlit dashboard
.\run.ps1 run
```

### Linux / macOS
```bash
make setup
make test
make run
```

---

## 4. Enterprise Cloud Deployment Strategies

### Option A: Serverless Cloud Run (Google Cloud Platform) — *Recommended for Cost-Efficiency*
Google Cloud Run provides auto-scaling container execution with zero cost when inactive during school holidays and summer breaks.

1. **Build and Submit Container Image**:
   ```bash
   gcloud builds submit --tag gcr.io/district-analytics-prod/early-warning-app:latest
   ```

2. **Deploy Service**:
   ```bash
   gcloud run deploy early-warning-system \
     --image gcr.io/district-analytics-prod/early-warning-app:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated=false \
     --memory 2Gi \
     --cpu 2 \
     --min-instances 0 \
     --max-instances 5 \
     --port 8501
   ```

3. **Configure District Single Sign-On (SSO)**:
   - Restrict access through Google Cloud Identity-Aware Proxy (IAP) integrated with Google Workspace for Education.

---

### Option B: AWS ECS Fargate (Amazon Web Services) — *Enterprise District Infrastructure*
For districts operating on AWS EdTech GovCloud environments:

1. **Push Image to Amazon ECR**:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com
   docker tag disengagement-early-warning:latest <account_id>.dkr.ecr.us-east-1.amazonaws.com/early-warning:latest
   docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/early-warning:latest
   ```

2. **Task Definition Specifications**:
   - Launch type: `FARGATE`
   - CPU: `1024` (1 vCPU)
   - Memory: `2048` (2 GB RAM)
   - Container Port: `8501`
   - Health Check: `CMD-SHELL, curl -f http://localhost:8501/_stcore/health || exit 1`

3. **Load Balancer & Networking**:
   - Application Load Balancer (ALB) terminating TLS 1.3 with an AWS Certificate Manager (ACM) wild-card certificate.
   - Enforce district WAF rules blocking non-district IP CIDR blocks.

---

## 5. FERPA, Privacy & Security Hardening Architecture

Educational deployments must strictly comply with the **Family Educational Rights and Privacy Act (FERPA)**, **COPPA**, and state student privacy regulations.

| Security Dimension | Technical Implementation | Operational Rationale |
| :--- | :--- | :--- |
| **In-Transit Encryption** | TLS 1.3 mandatory; all HTTP traffic redirected to HTTPS. | Prevents packet sniffing of student identifiers or academic marks across campus Wi-Fi networks. |
| **At-Rest Encryption** | AES-256 encrypted volumes for cached data and SQLite/PostgreSQL stores. | Guarantees data security in the event of hardware decommissioning or storage migration. |
| **Pseudonymization Gate** | Raw SIS IDs (`STU_0042`) hashed with district-held cryptographic salt before feature engineering. | Data science analysts and cloud logs never see real student names, addresses, or demographics. |
| **Role-Based Access Control (RBAC)** | Strict partition between Counselor View and Student View: | Counselors access risk scores and local SHAP explanations; students access strengths and self-advocacy habits without exposure to administrative risk probabilities. |
| **Non-Punitive Data Retention** | Advisory flags and risk scores are ephemeral decision-support artifacts deleted every 180 days. | Disengagement indicators NEVER write to permanent disciplinary records, official transcripts, or external student files. |

---

## 6. MLOps Automated Maintenance & Continuous Ingestion

### 6.1 Weekly Ingestion Pipeline Cron Job
To synchronize weekly data from OneRoster, Canvas, and PowerSchool:
```bash
# Add to crontab on analytics worker server (Runs every Monday at 04:00 AM)
0 4 * * 1 cd /app && python -m src.connectors.pipeline --sync-all --validate
```

### 6.2 Monthly Drift Supervision
Run the automated longitudinal drift monitor to compute feature PSI and KS statistics:
```bash
python src/drift_monitor.py
```
- If mean cohort $PSI \ge 0.10$, an advisory alert is dispatched to the district data science supervisor.
- If mean cohort $PSI \ge 0.25$, an automated retraining workflow is triggered, expanding prediction intervals to protect transfer students.

### 6.3 Semesterly Retraining Runbook
Every June and December:
1. Export the trailing 3 semesters of verified cohort data.
2. Execute temporal holdout backtesting: `python src/backtest.py`.
3. Verify that $ECE \le 0.050$, Counselor Recall $\ge 80\%$, and Precision $\ge 90\%$.
4. Have the District Academic Counseling Lead and Data Privacy Officer review the before-and-after synthesis table.
5. Deploy updated model weights to the container registry with automated canary routing.

---

## 7. Operational Troubleshooting

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| **Container returns 502 Bad Gateway** | Streamlit process taking $> 15s$ to compile tree models on low-CPU VM. | Increase Docker healthcheck `start_period` to `30s` and assign at least 1.0 dedicated vCPU. |
| **`InformationBarrierError` in logs** | Upstream SIS connector query included ground-truth audit columns. | Ensure `GROUND_TRUTH_COLUMNS` are excluded from the connector query payload. |
| **High False-Alarm Rate on Transfer Students** | Sparse historical data ($< 4$ weeks). | Verify `uncertainty_est.compute_instance_uncertainty()` is called with accurate `weeks_available`. |
