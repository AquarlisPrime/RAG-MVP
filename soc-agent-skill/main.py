!pip install langchain gradio pandas numpy requests

logs = """
Mar 12 10:22:41 sshd Failed password for root from 192.168.1.45
Mar 12 10:22:43 sshd Failed password for root from 192.168.1.45
Mar 12 10:22:45 sshd Failed password for root from 192.168.1.45
Mar 12 10:23:10 sshd Accepted password for user from 10.0.0.2
"""

import re
from collections import Counter
import gradio as gr

# Extract IP addresses
def parse_logs(log_text):
    pattern = r"from (\d+\.\d+\.\d+\.\d+)"
    ips = re.findall(pattern, log_text)
    return ips


# Detect brute force attempts
def detect_bruteforce(log_text):
    ips = parse_logs(log_text)
    counts = Counter(ips)

    suspicious = {ip: count for ip, count in counts.items() if count >= 3}

    return suspicious


# Simulated threat intel lookup
def ip_reputation(ip):
    malicious_ips = ["192.168.1.45"]

    if ip in malicious_ips:
        return "Malicious"
    else:
        return "Unknown"


# Generate SOC report
def generate_report(log_text):

    threats = detect_bruteforce(log_text)

    if not threats:
        return "✅ No threat detected"

    report = "⚠ SECURITY ALERT\n\n"

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


# Gradio Interface
def analyze(log_text):
    return generate_report(log_text)


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=15, placeholder="Paste system logs here..."),
    outputs="textbox",
    title="AI SOC Log Analyzer Agent",
    description="Detects brute-force login attempts and suspicious IP activity from security logs."
)

demo.launch()
