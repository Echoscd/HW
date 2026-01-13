import time
import requests
from prometheus_client.parser import text_string_to_metric_families
from rich.console import Console
from rich.table import Table

def fetch_metrics(url: str = "http://localhost:8000/metrics"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch metrics: {e}")

def parse_metrics(raw_metrics: str):
    return list(text_string_to_metric_families(raw_metrics))

def build_table(metric_families):
    table = Table(title="vLLM Metrics Monitor")
    table.add_column("Metric Name", style="cyan")
    table.add_column("Labels", style="magenta")
    table.add_column("Value", style="green")
    table.add_column("Type", style="yellow")

    for family in metric_families:
        for sample in family.samples:
            labels_str = ", ".join(f"{k}={v}" for k, v in sample.labels.items())
            table.add_row(sample.name, labels_str, str(sample.value), family.type)

    return table

def run_monitor(url: str = "http://localhost:8066/metrics", refresh_interval: float = 1.0):
    console = Console()
    while True:
        try:
            raw = fetch_metrics(url)
            families = parse_metrics(raw)
            table = build_table(families)

            console.print(table)
            console.print("-" * 80)  # Divider for clarity
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        time.sleep(refresh_interval)

if __name__ == "__main__":
    port = 8066
    url = f"http://localhost:{port}/metrics"
    run_monitor(url=url, refresh_interval=1.0)
