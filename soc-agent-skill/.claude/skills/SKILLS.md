---
name: log-analyzer
description: Analyzes security logs for anomalies and maps findings to MITRE ATT&CK. Use when a user provides raw log data (Syslog, Auth, PCAP) or asks for an incident triage.
---

# Log Analysis Workflow

## Step 1: Normalization
Execute `python3 scripts/parser.py` on the provided log input to convert it into a structured JSON format.

## Step 2: Anomaly Detection
Scan the normalized data for the following indicators:
- Repeated failed logins (Brute Force)
- Unusual outbound traffic to unknown IPs
- Privilege escalation commands (e.g., unauthorized `sudo` usage)

## Step 3: Threat Mapping
Cross-reference findings with `references/mitre.json`.

## Step 4: Reporting
Generate a summary. If critical threats are found, trigger the `incident-report` skill.
