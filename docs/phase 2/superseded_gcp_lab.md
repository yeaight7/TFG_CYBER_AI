# Private Lab Setup (this is not what wass finally used for Phase 2, this was the original approach)

This guide describes a minimal isolated lab for Phase 2 traffic generation and offline evaluation.

Although the examples below use GCP terminology, the topology is provider-agnostic.

## Recommended Topology

### Minimal Two-VM Lab

| VM | IP | Role |
|---|---|---|
| `attacker` | `<ATTACKER_PRIVATE_IP>` | Generates benign and malicious traffic |
| `defender` | `<DEFENDER_PRIVATE_IP>` | Hosts target services, captures traffic, runs offline inference |

## Safety Requirements

- use a private VPC only
- do not expose attacker or defender publicly
- allow SSH only from a controlled source IP or through a bastion/IAP path
- keep attack traffic inside the lab
- do not store credentials in the repository

## GCP Example

### 1. Create the Network

```bash
gcloud compute networks create tfg-lab-vpc \
  --subnet-mode=custom

gcloud compute networks subnets create tfg-lab-subnet \
  --network=tfg-lab-vpc \
  --region=europe-west1 \
  --range=10.0.0.0/24
```

### 2. Create Firewall Rules

```bash
gcloud compute firewall-rules create tfg-allow-ssh \
  --network=tfg-lab-vpc \
  --allow=tcp:22 \
  --source-ranges=<YOUR_PUBLIC_IP>/32

gcloud compute firewall-rules create tfg-allow-internal \
  --network=tfg-lab-vpc \
  --allow=tcp,udp,icmp \
  --source-ranges=10.0.0.0/24
```

If you enforce strict egress control, document the exact rules you apply and ensure the lab still supports package installation through your approved path.

### 3. Create the VMs

```bash
gcloud compute instances create tfg-attacker \
  --zone=europe-west1-b \
  --machine-type=e2-medium \
  --image-family=kali-rolling \
  --image-project=kali-linux-cloud \
  --network-interface=subnet=tfg-lab-subnet,private-network-ip=<ATTACKER_PRIVATE_IP>,no-address \
  --boot-disk-size=30GB

gcloud compute instances create tfg-defender \
  --zone=europe-west1-b \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --network-interface=subnet=tfg-lab-subnet,private-network-ip=<DEFENDER_PRIVATE_IP>,no-address \
  --boot-disk-size=50GB
```

## Defender Setup

Install the minimum tooling needed for Phase 2:

```bash
sudo apt update
sudo apt install -y docker.io tcpdump tshark python3-pip default-jdk
pip install -r requirements.txt
```

Optional example services:

```bash
docker run -d --name nginx -p 80:80 nginx:latest
docker run -d --name ssh -p 2222:22 rastasheep/ubuntu-sshd
docker run -d --name ftp -p 21:21 stilliard/pure-ftpd
docker run -d --name mysql -p 3306:3306 -e MYSQL_ROOT_PASSWORD=test mysql:8
```

## Attacker Setup

Update the Kali VM and verify your tools are available:

```bash
sudo apt update && sudo apt upgrade -y
```

Typical tools:

- `curl`
- `wget`
- `nmap`
- `hping3`
- `hydra`
- `sqlmap`

## Traffic Generation Examples

### Benign

```bash
for i in $(seq 1 100); do
  curl -s http://<DEFENDER_PRIVATE_IP>/ > /dev/null
  sleep 0.5
done
```

### Scan

```bash
nmap -sS -T4 <DEFENDER_PRIVATE_IP>
```

### Short SYN Flood Burst

```bash
sudo hping3 -S --flood -V -p 80 <DEFENDER_PRIVATE_IP> &
sleep 10
kill %1
```

## Ground-Truth Logging

Keep a separate traffic log such as:

```json
[
  { "start": "2026-03-01T10:00:00", "end": "2026-03-01T10:05:00", "type": "benign_http", "label": 0 },
  { "start": "2026-03-01T10:05:00", "end": "2026-03-01T10:06:00", "type": "nmap_scan", "label": 1 }
]
```

The model should never be treated as the source of truth for the labels.

## Capture and Feature Extraction

Capture:

```bash
sudo tcpdump -i eth0 -w /data/captures/session_$(date +%Y%m%d_%H%M%S).pcap
```

Extract:

```bash
java -jar CICFlowMeter.jar -i /data/captures/ -o /data/flows/
```

Then use the maintained inference pipeline:

- `scripts/predict_real_traffic_v2.py`

## Shutdown and Cleanup

Stop the instances when not in use:

```bash
gcloud compute instances stop tfg-attacker tfg-defender --zone=europe-west1-b
```

Or delete the whole lab:

```bash
gcloud compute instances delete tfg-attacker tfg-defender --zone=europe-west1-b --quiet
gcloud compute firewall-rules delete tfg-allow-ssh tfg-allow-internal --quiet
gcloud compute networks subnets delete tfg-lab-subnet --region=europe-west1 --quiet
gcloud compute networks delete tfg-lab-vpc --quiet
```

## Adapting to Other Providers

| Concern | AWS | Azure | Local virtualisation |
|---|---|---|---|
| Private network | VPC + private subnet | VNet + subnet | host-only or isolated bridge |
| Controlled SSH | SSM or bastion | Bastion | host-only SSH |
| Teardown | terminate instances | deallocate/delete VMs | destroy or stop VMs |
