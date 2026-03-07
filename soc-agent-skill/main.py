!pip install langchain gradio pandas numpy requests

logs = """
Mar 12 10:22:41 sshd Failed password for root from 192.168.1.45
Mar 12 10:22:43 sshd Failed password for root from 192.168.1.45
Mar 12 10:22:45 sshd Failed password for root from 192.168.1.45
Mar 12 10:23:10 sshd Accepted password for user from 10.0.0.2
"""
import re

def parse_logs(log_text):
    pattern = r"from (\d+\.\d+\.\d+\.\d+)"
    ips = re.findall(pattern, log_text)
    return ips

from collections import Counter

def detect_bruteforce(log_text):
    ips = parse_logs(log_text)
    counts = Counter(ips)

    suspicious = {ip:count for ip,count in counts.items() if count >=3}

    return suspicious

def ip_reputation(ip):
    malicious_ips = ["192.168.1.45"]

    if ip in malicious_ips:
        return "Malicious"
    else:
        return "Unknown"

  def generate_report(log_text):

    threats = detect_bruteforce(log_text)

    if not threats:
        return "No threat detected"

    report = "⚠ Security Alert\n\n"

    for ip, attempts in threats.items():
        rep = ip_reputation(ip)

        report += f"""
Threat: Brute Force Login
Source IP: {ip}
Attempts: {attempts}
Reputation: {rep}
Risk Level: HIGH

Recommendation:
Block IP and investigate authentication logs
"""

    return report

print(generate_report(logs))

import gradio as gr

def analyze(log_text):
    return generate_report(log_text)

demo = gr.Interface(
    fn=analyze,
    inputs="textbox",
    outputs="textbox",
    title="AI SOC Log Analyzer Agent"
)

demo.launch()
