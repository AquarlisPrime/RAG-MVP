def ip_reputation(ip):
    """
    Simple threat intelligence lookup.
    Replace with real APIs like:
    - VirusTotal
    - AbuseIPDB
    """

    malicious_ips = [
        "192.168.1.45",
        "185.220.101.1"
    ]

    if ip in malicious_ips:
        return "Malicious"
    else:
        return "Unknown"


def enrich_ips(ip_list):

    results = {}

    for ip in ip_list:
        results[ip] = ip_reputation(ip)

    return results


if __name__ == "__main__":

    test_ips = ["192.168.1.45", "10.0.0.2"]

    print(enrich_ips(test_ips))
