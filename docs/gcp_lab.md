# Private Lab Setup (Cloud-Agnostic, GCP Example)

Instructions for deploying a minimal private lab to generate and evaluate network traffic for Phase 2. Although the examples use **Google Cloud Platform (GCP)**, the topology is cloud-agnostic and works on any provider (AWS, Azure, local VMs, etc.).

---

## Topology

### Minimal (2-VM)

```
┌─────────────────────────────── Private VPC (10.0.0.0/24) ──────────────────────────────┐
│                                                                                        │
│   ┌──────────────────┐          eth0 ←→ eth0          ┌──────────────────────────┐     │
│   │  attacker VM     │ ──────────────────────────────→│  defender VM             │     │
│   │  Kali Linux      │                                │  Ubuntu 22.04            │     │
│   │                  │  generates benign + attack     │  - Docker (nginx, ssh,   │     │
│   │  hping3, nmap,   │  traffic towards defender      │    ftp, mysql targets)   │     │
│   │  hydra, sqlmap,  │                                │  - tcpdump / tshark      │     │
│   │  curl, wget      │                                │  - CICFlowMeter          │     │
│   │                  │                                │  - Python + QRDQN model  │     │
│   └──────────────────┘                                └──────────────────────────┘     │
│        10.0.0.10                                              10.0.0.20                │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                    SSH only from
                                    your IP (bastion)
```

### Optional 3-VM Topology

Add a **monitor** VM that passively mirrors traffic for analysis:

| VM | IP | Role |
|----|----|------|
| attacker | 10.0.0.10 | Traffic generation (Kali) |
| defender | 10.0.0.20 | Target services + capture + agent |
| monitor | 10.0.0.30 | Passive tap, ELK/Grafana dashboards (optional) |

---

## Safety Guardrails

> **These rules are mandatory.** Violating them may expose external systems or violate laws.

| # | Rule | How to enforce |
|---|------|----------------|
| 1 | **Private VPC only** | Create a VPC with no default internet gateway; no public IPs on attacker/defender |
| 2 | **SSH from your IP only** | Firewall rule: allow TCP/22 only from `<YOUR_PUBLIC_IP>/32` |
| 3 | **No scanning outside the lab** | All attack tools target `10.0.0.0/24` only; no DNS resolution of external hosts |
| 4 | **Ephemeral VMs** | Stop or delete VMs when not in use (`gcloud compute instances stop`) |
| 5 | **No credentials in code** | Use GCP metadata server or env vars; never commit keys to the repo |
| 6 | **Firewall deny-all egress** | Default egress rule: DENY all; allow only internal + apt mirrors if needed |

---

## GCP Setup (Step-by-Step)

### 1. Create VPC & Subnet

```bash
gcloud compute networks create tfg-lab-vpc \
    --subnet-mode=custom

gcloud compute networks subnets create tfg-lab-subnet \
    --network=tfg-lab-vpc \
    --region=europe-west1 \
    --range=10.0.0.0/24
```

### 2. Firewall Rules

```bash
# Allow SSH from your IP only
gcloud compute firewall-rules create tfg-allow-ssh \
    --network=tfg-lab-vpc \
    --allow=tcp:22 \
    --source-ranges=<YOUR_PUBLIC_IP>/32

# Allow all internal traffic within the VPC
gcloud compute firewall-rules create tfg-allow-internal \
    --network=tfg-lab-vpc \
    --allow=tcp,udp,icmp \
    --source-ranges=10.0.0.0/24

# Deny all egress (override default)
gcloud compute firewall-rules create tfg-deny-egress \
    --network=tfg-lab-vpc \
    --direction=EGRESS \
    --action=DENY \
    --rules=all \
    --destination-ranges=0.0.0.0/0 \
    --priority=1000

# Allow egress to internal only
gcloud compute firewall-rules create tfg-allow-egress-internal \
    --network=tfg-lab-vpc \
    --direction=EGRESS \
    --action=ALLOW \
    --rules=all \
    --destination-ranges=10.0.0.0/24 \
    --priority=900
```

### 3. Create VMs

```bash
# Attacker (Kali)
gcloud compute instances create tfg-attacker \
    --zone=europe-west1-b \
    --machine-type=e2-medium \
    --image-family=kali-rolling \
    --image-project=kali-linux-cloud \
    --network-interface=subnet=tfg-lab-subnet,private-network-ip=10.0.0.10,no-address \
    --boot-disk-size=30GB

# Defender (Ubuntu)
gcloud compute instances create tfg-defender \
    --zone=europe-west1-b \
    --machine-type=e2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --network-interface=subnet=tfg-lab-subnet,private-network-ip=10.0.0.20,no-address \
    --boot-disk-size=50GB
```

> **Note**: `no-address` ensures no public IP. SSH via IAP tunnel:
> ```bash
> gcloud compute ssh tfg-defender --zone=europe-west1-b --tunnel-through-iap
> ```

### 4. Install Software on Defender

```bash
sudo apt update && sudo apt install -y docker.io tcpdump tshark python3-pip default-jdk
pip install -r requirements.txt  # from repo root

# Start target services in Docker
docker run -d --name nginx  -p 80:80   nginx:latest
docker run -d --name ssh    -p 2222:22 rastasheep/ubuntu-sshd
docker run -d --name ftp    -p 21:21   stilliard/pure-ftpd
docker run -d --name mysql  -p 3306:3306 -e MYSQL_ROOT_PASSWORD=test mysql:8
```

### 5. Install Software on Attacker

Kali Linux comes pre-installed with `nmap`, `hping3`, `hydra`, `sqlmap`, `nikto`, `curl`, `wget`, etc. Update tools:

```bash
sudo apt update && sudo apt upgrade -y
```

---

## Generating Traffic

### Benign Traffic (from attacker → defender)

```bash
# HTTP browsing
for i in $(seq 1 100); do curl -s http://10.0.0.20/ > /dev/null; sleep 0.5; done

# File downloads
wget -q http://10.0.0.20/index.html -O /dev/null

# SSH sessions
ssh -o StrictHostKeyChecking=no user@10.0.0.20 -p 2222 'ls; whoami; uptime'
```

### Attack Traffic (from attacker → defender)

```bash
# Port scan
nmap -sS -T4 10.0.0.20

# DDoS (SYN flood) — short burst only!
sudo hping3 -S --flood -V -p 80 10.0.0.20 &
sleep 10 && kill %1

# Brute force SSH
hydra -l root -P /usr/share/wordlists/rockyou.txt 10.0.0.20 ssh -s 2222 -t 4

# SQL injection
sqlmap -u "http://10.0.0.20/vuln?id=1" --batch --level=1
```

### Label Log

Create a simple JSON log to record ground-truth:

```json
[
  { "start": "2026-03-01T10:00:00", "end": "2026-03-01T10:05:00", "type": "benign_http", "label": 0 },
  { "start": "2026-03-01T10:05:00", "end": "2026-03-01T10:06:00", "type": "nmap_scan",   "label": 1 },
  { "start": "2026-03-01T10:06:00", "end": "2026-03-01T10:10:00", "type": "benign_ssh",   "label": 0 },
  { "start": "2026-03-01T10:10:00", "end": "2026-03-01T10:11:00", "type": "syn_flood",    "label": 1 }
]
```

---

## Capture & Feature Extraction

```bash
# On defender: capture while traffic is generated
sudo tcpdump -i eth0 -w /data/captures/session_$(date +%Y%m%d_%H%M%S).pcap

# After capture: extract flows
java -jar CICFlowMeter.jar -i /data/captures/ -o /data/flows/
```

Then use `canonical_schema.py` to map to the 152-dim vector (see [`docs/phase2_plan.md`](phase2_plan.md) for details).

---

## Teardown

```bash
# Stop VMs (preserves disks, no compute charges)
gcloud compute instances stop tfg-attacker tfg-defender --zone=europe-west1-b

# Or delete everything
gcloud compute instances delete tfg-attacker tfg-defender --zone=europe-west1-b --quiet
gcloud compute firewall-rules delete tfg-allow-ssh tfg-allow-internal tfg-deny-egress tfg-allow-egress-internal --quiet
gcloud compute networks subnets delete tfg-lab-subnet --region=europe-west1 --quiet
gcloud compute networks delete tfg-lab-vpc --quiet
```

---

## Adapting for Other Providers

| Step | AWS Equivalent | Azure Equivalent | Local (VirtualBox/libvirt) |
|------|---------------|------------------|---------------------------|
| VPC | VPC + private subnet | VNet + subnet | Host-only network |
| Firewall | Security Groups | NSG rules | iptables on host |
| VMs | EC2 instances (no public IP) | Azure VMs (no PIP) | Local VMs |
| SSH tunnel | SSM Session Manager | Azure Bastion | Direct SSH on host-only |
| Teardown | Terminate instances | Deallocate VMs | virsh destroy / VBoxManage |
