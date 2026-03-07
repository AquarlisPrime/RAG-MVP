# Security Policy

This repository contains a **research prototype for an AI SOC log analysis skill**.

The system analyzes authentication logs to detect brute-force attacks
and suspicious login behavior.

---

# Threat Model

Potential risks include:

### Prompt Injection

Attackers may manipulate log entries to influence AI analysis.

Example:

Malicious text embedded in logs that instructs the AI to ignore attacks.

Mitigation:

- sanitize log input
- isolate skill execution

---

### Sensitive Data Exposure

Logs may contain:

- IP addresses
- usernames
- infrastructure details

Mitigation:

- anonymize logs
- restrict access

---

### Incorrect Threat Intelligence

Threat intel sources may produce false positives.

Mitigation:

- combine multiple sources
- require analyst verification

---

# Responsible Use

This project is intended for:

- research
- experimentation
- cybersecurity education

It should not be deployed in production SOC environments
without additional validation and security review.
