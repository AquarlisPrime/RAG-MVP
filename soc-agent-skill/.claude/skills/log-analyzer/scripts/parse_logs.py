import re
from collections import Counter


def extract_ips(log_text):
    """
    Extract IP addresses from logs
    """
    pattern = r"from (\d+\.\d+\.\d+\.\d+)"
    return re.findall(pattern, log_text)


def detect_bruteforce(log_text, threshold=3):
    """
    Detect brute force attacks based on repeated login attempts
    """
    ips = extract_ips(log_text)

    counts = Counter(ips)

    suspicious = {
        ip: count for ip, count in counts.items()
        if count >= threshold
    }

    return suspicious


if __name__ == "__main__":

    sample_logs = """
    Mar 12 10:22:41 sshd Failed password for root from 192.168.1.45
    Mar 12 10:22:43 sshd Failed password for root from 192.168.1.45
    Mar 12 10:22:45 sshd Failed password for root from 192.168.1.45
    """

    print(detect_bruteforce(sample_logs))
