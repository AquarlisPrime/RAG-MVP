name: log-analyzer
description: SOC agent skill for detecting brute-force login attacks from system logs
version: 1.0
author: AquarlisPrime
tools:
  - parse_logs
  - check_threat_intel
tags:
  - cybersecurity
  - soc
  - threat-detection
  - incident-response

---

# Log Analyzer Skill

This skill enables an AI security agent to analyze authentication logs
and detect brute-force login attacks.

It is designed to support **SOC analysts and AI agents** by automating
initial investigation tasks.

---

# Investigation Workflow

### Step 1 – Parse Logs

Extract:

- Source IP addresses
- Authentication status
- Login timestamps

### Step 2 – Detect Suspicious Activity

Flag patterns such as:

- ≥3 failed login attempts from same IP
- rapid authentication failures
- unusual login sources

### Step 3 – Threat Intelligence Lookup

Check suspicious IPs against known malicious indicators.

### Step 4 – ATT&CK Mapping

Map the detected activity to MITRE ATT&CK techniques.

Example:

| Behavior | Technique |
|--------|--------|
Multiple login attempts | T1110 (Brute Force)

### Step 5 – Generate SOC Report

The skill should output:

- suspicious IP addresses
- attack type
- risk level
- recommended response
