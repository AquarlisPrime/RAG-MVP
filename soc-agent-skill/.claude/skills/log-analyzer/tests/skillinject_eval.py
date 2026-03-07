"""
Simple evaluation for the log analyzer skill.
"""

from collections import Counter


def evaluate_detection(ip_list):

    counts = Counter(ip_list)

    suspicious = {
        ip: count for ip, count in counts.items()
        if count >= 3
    }

    if suspicious:
        return "PASS"

    return "FAIL"


if __name__ == "__main__":

    test_ips = [
        "192.168.1.45",
        "192.168.1.45",
        "192.168.1.45",
        "10.0.0.2"
    ]

    result = evaluate_detection(test_ips)

    print("Skill evaluation:", result)
