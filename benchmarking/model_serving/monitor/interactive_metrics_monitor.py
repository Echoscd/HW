import time
from collections import deque
from typing import Dict, List, Tuple

import requests
from prometheus_client.parser import text_string_to_metric_families
from textual.app import App
from textual.reactive import reactive
from textual.widgets import Footer, Header, TabbedContent, TabPane, DataTable
from textual_plotext import PlotextPlot

DEFAULT_URL = "http://localhost:8066/metrics"
MAX_HISTORY = 300  # ~5 minutes at 1s interval
POLL_INTERVAL = 2.0  # seconds

class MetricsCollector:
    def __init__(self, max_history: int = MAX_HISTORY):
        self.history: Dict[str, deque[Tuple[float, float]]] = {}
        self.prev_counters: Dict[str, float] = {}
        self.max_history = max_history
        self.baseline: Dict[str, float] = {}  # Store baseline values for counters/histograms

    def reset_baseline(self, metrics: Dict[str, float]):
        """Reset baseline values for counters and histogram sums/counts."""
        self.baseline = {k: v for k, v in metrics.items() if k.endswith('_total') or k.endswith('_sum') or k.endswith('_count')}
        self.history.clear()  # Clear history to start fresh
        self.prev_counters.clear()

    def update(self, metrics: Dict[str, float]):
        now = time.time()
        # Compute delta metrics relative to baseline
        delta_metrics = {}
        for key, value in metrics.items():
            if key in self.baseline:
                delta_metrics[key] = value - self.baseline.get(key, 0.0)
            else:
                delta_metrics[key] = value

        # Store delta metrics in history
        for key, value in delta_metrics.items():
            if key not in self.history:
                self.history[key] = deque(maxlen=self.max_history)
            self.history[key].append((now, value))

        # Compute rates for counters
        rate_mappings = {
            'vllm:prompt_tokens_rate': 'vllm:prompt_tokens_total',
            'vllm:generation_tokens_rate': 'vllm:generation_tokens_total'
        }
        for rate_key, base_key in rate_mappings.items():
            if rate_key in delta_metrics:
                if base_key in self.prev_counters and base_key in delta_metrics:
                    dt = POLL_INTERVAL
                    rate = (delta_metrics[base_key] - self.prev_counters[base_key]) / dt
                    self.history[rate_key].append((now, max(rate, 0.0)))
                self.prev_counters[base_key] = delta_metrics.get(base_key, 0.0)

def fetch_metrics(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"# Error fetching metrics: {e}"

def parse_metrics(raw: str) -> Dict:
    if raw.startswith("# Error"):
        return {'gauges': {}, 'histograms': {}}
    
    metrics = {}
    histograms = {}
    families = text_string_to_metric_families(raw)
    for family in families:
        if not family.name.startswith('vllm:'):
            continue
        hist_name = family.name.replace('vllm:', '')
        if family.type == 'histogram':
            buckets = []
            cumulative_counts = []
            hist_sum = 0.0
            hist_count = 0.0
            for sample in family.samples:
                if '_bucket' in sample.name:
                    le = sample.labels.get('le', '')
                    try:
                        le_val = float(le)
                    except ValueError:
                        le_val = float('inf')
                    buckets.append((le_val, le))
                    cumulative_counts.append(sample.value)
                elif '_sum' in sample.name:
                    hist_sum = sample.value
                elif '_count' in sample.name:
                    hist_count = sample.value
            sorted_buckets = sorted(zip(buckets, cumulative_counts), key=lambda x: x[0][0])
            buckets = [b[1] for b, _ in sorted_buckets]
            cumulative_counts = [c for _, c in sorted_buckets]
            histograms[hist_name] = {'buckets': buckets, 'cumulative_counts': cumulative_counts, 'sum': hist_sum, 'count': hist_count}
        else:
            for sample in family.samples:
                if sample.name == family.name:
                    metrics[sample.name] = sample.value
                elif '_sum' in sample.name:
                    metrics[sample.name] = sample.value
                elif '_count' in sample.name:
                    metrics[sample.name] = sample.value

    for hist_name in histograms:
        h = histograms[hist_name]
        if h['count'] > 0:
            metrics[f'vllm:{hist_name}_mean'] = h['sum'] / h['count']

    keys = [
        'vllm:num_requests_running',
        'vllm:num_requests_waiting',
        'vllm:kv_cache_usage_perc',
        'vllm:prompt_tokens_total',
        'vllm:generation_tokens_total',
        'vllm:time_to_first_token_seconds_mean',
        'vllm:time_per_output_token_seconds_mean',
        'vllm:e2e_request_latency_seconds_mean',
        'vllm:request_queue_time_seconds_mean',
        'vllm:request_prefill_time_seconds_mean',
        'vllm:request_decode_time_seconds_mean',
    ]
    gauges = {k: metrics.get(k, 0.0) for k in keys}
    gauges['vllm:prompt_tokens_rate'] = 0.0
    gauges['vllm:generation_tokens_rate'] = 0.0
    
    return {'gauges': gauges, 'histograms': histograms}

class MetricsPlot(PlotextPlot):
    title = reactive("")
    series: List[str] = reactive([])

    def on_mount(self) -> None:
        self.plt.title(self.title)
        self.plt.xlabel("Time (s)")
        self.plt.ylabel("Value")

    def update_plot(self, history: Dict[str, deque[Tuple[float, float]]]) -> None:
        self.plt.clear_figure()
        if not self.series:
            return
        
        min_t = float('inf')
        for key in self.series:
            if key in history and history[key]:
                ts, vs = zip(*history[key])
                min_t = min(min_t, min(ts))
                self.plt.plot([t - min_t for t in ts], vs, label=key.split(':')[-1])
        
        self.plt.title(self.title)
        self.plt.xlabel("Time (s)")
        self.plt.ylabel("Value")
        self.plt.show_grid()
        self.refresh()

class HistogramPlot(PlotextPlot):
    title = reactive("")
    hist_key = reactive("")

    def on_mount(self) -> None:
        self.plt.title(self.title)
        self.plt.xlabel("Bucket")
        self.plt.ylabel("Count")

    def update_plot(self, hist_data: Dict) -> None:
        self.plt.clear_figure()
        if not hist_data:
            return
        buckets = hist_data['buckets']
        cumulative_counts = hist_data['cumulative_counts']
        diffs = [cumulative_counts[0]] + [cumulative_counts[i] - cumulative_counts[i-1] for i in range(1, len(cumulative_counts))]
        self.plt.bar([str(b) for b in buckets], diffs, label="Count")
        self.plt.title(self.title)
        self.plt.xlabel("Bucket")
        self.plt.ylabel("Count")
        self.plt.show_grid()
        self.refresh()

class VLLMMetricsApp(App):
    BINDINGS = [("q", "quit", "Quit"), ("r", "reset_metrics", "Reset Metrics")]

    def __init__(self, url: str = DEFAULT_URL):
        super().__init__()
        self.url = url
        self.collector = MetricsCollector()

    def compose(self):
        yield Header()
        with TabbedContent():
            with TabPane("Raw Metrics", id="raw-metrics"):
                table = DataTable(id="metrics-table")
                table.add_columns("Metric", "Value")
                yield table
            with TabPane("Scheduler", id="scheduler"):
                plot = MetricsPlot(id="plot-scheduler")
                plot.title = "Scheduler States"
                plot.series = ["vllm:num_requests_running", "vllm:num_requests_waiting"]
                yield plot
            with TabPane("KV Cache", id="kv-cache"):
                plot = MetricsPlot(id="plot-kv")
                plot.title = "KV Cache Usage (%)"
                plot.series = ["vllm:kv_cache_usage_perc"]
                yield plot
            with TabPane("Token Rates", id="tokens"):
                plot = MetricsPlot(id="plot-tokens")
                plot.title = "Token Throughput (tokens/s)"
                plot.series = ["vllm:prompt_tokens_rate", "vllm:generation_tokens_rate"]
                yield plot
            with TabPane("Latencies", id="latencies"):
                plot = MetricsPlot(id="plot-latencies")
                plot.title = "Latencies (s)"
                plot.series = [
                    "vllm:time_to_first_token_seconds_mean",
                    "vllm:time_per_output_token_seconds_mean",
                    "vllm:e2e_request_latency_seconds_mean"
                ]
                yield plot
            with TabPane("Queue/Prefill/Decode", id="queue-prefill-decode"):
                plot = MetricsPlot(id="plot-queue-prefill-decode")
                plot.title = "Queue/Prefill/Decode Means (s)"
                plot.series = [
                    "vllm:request_queue_time_seconds_mean",
                    "vllm:request_prefill_time_seconds_mean",
                    "vllm:request_decode_time_seconds_mean"
                ]
                yield plot
            with TabPane("TTFT Hist", id="ttft-hist"):
                plot = HistogramPlot(id="hist-ttft")
                plot.title = "Time to First Token Distribution (s)"
                plot.hist_key = "time_to_first_token_seconds"
                yield plot
            with TabPane("TPOT Hist", id="tpot-hist"):
                plot = HistogramPlot(id="hist-tpot")
                plot.title = "Time per Output Token Distribution (s)"
                plot.hist_key = "time_per_output_token_seconds"
                yield plot
            with TabPane("E2E Latency Hist", id="e2e-hist"):
                plot = HistogramPlot(id="hist-e2e")
                plot.title = "E2E Request Latency Distribution (s)"
                plot.hist_key = "e2e_request_latency_seconds"
                yield plot
            with TabPane("Prompt Tokens Hist", id="prompt-hist"):
                plot = HistogramPlot(id="hist-prompt")
                plot.title = "Prompt Tokens Distribution"
                plot.hist_key = "request_prompt_tokens"
                yield plot
            with TabPane("Generation Tokens Hist", id="gen-hist"):
                plot = HistogramPlot(id="hist-gen")
                plot.title = "Generation Tokens Distribution"
                plot.hist_key = "request_generation_tokens"
                yield plot
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(POLL_INTERVAL, self.update_metrics)

    async def update_metrics(self) -> None:
        raw = fetch_metrics(self.url)
        parsed = parse_metrics(raw)
        if 'gauges' in parsed:
            self.collector.update(parsed['gauges'])
        
        # Update line plots
        for plot_id in ["plot-scheduler", "plot-kv", "plot-tokens", "plot-latencies", "plot-queue-prefill-decode"]:
            try:
                plot = self.query_one(f"#{plot_id}", MetricsPlot)
                plot.update_plot(self.collector.history)
            except Exception as e:
                print(f"Error updating line plot {plot_id}: {e}")

        # Update histogram plots
        hist_ids = {
            "hist-ttft": "time_to_first_token_seconds",
            "hist-tpot": "time_per_output_token_seconds",
            "hist-e2e": "e2e_request_latency_seconds",
            "hist-prompt": "request_prompt_tokens",
            "hist-gen": "request_generation_tokens"
        }
        for hist_plot_id, hist_key in hist_ids.items():
            try:
                plot = self.query_one(f"#{hist_plot_id}", HistogramPlot)
                if hist_key in parsed.get('histograms', {}):
                    hist_data = parsed['histograms'][hist_key]
                    # Adjust histogram counts for baseline
                    if 'count' in self.collector.baseline and hist_data['count'] >= self.collector.baseline.get('count', 0.0):
                        hist_data['cumulative_counts'] = [c - self.collector.baseline.get('count', 0.0) for c in hist_data['cumulative_counts']]
                        hist_data['sum'] = hist_data['sum'] - self.collector.baseline.get('sum', 0.0)
                        hist_data['count'] = hist_data['count'] - self.collector.baseline.get('count', 0.0)
                    plot.update_plot(hist_data)
            except Exception as e:
                print(f"Error updating histogram plot {hist_plot_id}: {e}")

        # Update raw metrics table
        try:
            table = self.query_one("#metrics-table", DataTable)
            table.clear()
            for key, value in sorted(self.collector.history.items()):
                if self.collector.history[key]:  # Only show metrics with data
                    latest_value = self.collector.history[key][-1][1]  # Latest (time, value) tuple
                    table.add_row(key, f"{latest_value:.4f}")
        except Exception as e:
            print(f"Error updating metrics table: {e}")

    def action_reset_metrics(self) -> None:
        """Reset the collector's baseline for a new benchmark run."""
        raw = fetch_metrics(self.url)
        parsed = parse_metrics(raw)
        if 'gauges' in parsed:
            self.collector.reset_baseline(parsed['gauges'])
        print("Metrics baseline reset for new benchmark run.")

    def action_quit(self) -> None:
        self.app.exit()

if __name__ == "__main__":
    app = VLLMMetricsApp()
    app.run()