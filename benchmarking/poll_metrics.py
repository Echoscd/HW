import asyncio
import time
import json
import requests
from typing import AsyncGenerator, Dict, Any
from prometheus_client.parser import text_string_to_metric_families
from vllm.v1.metrics import reader as vllm_reader

class PollMetrics:
    """
    Async generator that polls at the /metrics endpoint,
    parse w with vLLM's v1 reader
    and yields JSON dict.
    """
    def __init__(self, metrics_url: str, interval: float = 1.0):
        self.metrics_url = metrics_url
        self.interval = interval

    def _get_raw(self) -> str:
        return requests.get(self.metrics_url, timeout=1).text

    def _parse(self, raw: str) -> Dict[str, Any]:
        families = list(text_string_to_metric_families(raw))
        parsed = []
        for metric in families:
            if not metric.name.startswith("vllm:"):
                continue
            # gauge
            if metric.type == "gauge":
                for s in vllm_reader._get_samples(metric):
                    parsed.append({
                        "type": "gauge",
                        "name": metric.name,
                        "labels": s.labels,
                        "value": s.value,
                    })
            # counter
            elif metric.type == "counter":
                samples = vllm_reader._get_samples(metric, "_total")
                if metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                    for labels, vec in vllm_reader._digest_num_accepted_by_pos_samples(samples):
                        parsed.append({
                            "type": "vector",
                            "name": metric.name,
                            "labels": labels,
                            "value": vec,
                        })
                else:
                    for s in samples:
                        parsed.append({
                            "type": "counter",
                            "name": metric.name,
                            "labels": s.labels,
                            "value": int(s.value),
                        })
            # histogram
            elif metric.type == "histogram":
                # we dont care about histogram atm
                continue
                # buckets = vllm_reader._get_samples(metric, "_bucket")
                # counts  = vllm_reader._get_samples(metric, "_count")
                # sums    = vllm_reader._get_samples(metric, "_sum")
                # for labels, bkt, cnt, sm in vllm_reader._digest_histogram(buckets, counts, sums):
                #     parsed.append({
                #         "type": "histogram",
                #         "name": metric.name,
                #         "labels": labels,
                #         "buckets": bkt,
                #         "count": cnt,
                #         "sum": sm,
                #     })
        # return as dict of metric-name -> list-of-entries
        out: Dict[str, Any] = {}
        for entry in parsed:
            out.setdefault(entry["name"], []).append(entry)
        return out

    async def poll(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async generator yielding a dict:
        {
            "timestamp": <UNIX epoch float>,
            "metrics": { metric_name: [ ... entries ... ] }
        }
        Poll every `self.interval` seconds.
        """
        try:
            while True:
                raw = self._get_raw()
                metrics = self._parse(raw)
                yield {
                    "timestamp": time.time(),
                    "metrics": metrics,
                }
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            return


async def run_fake_benchmark():
    # run benchmark code exampe
    await asyncio.sleep(6)  # e.g. run for 6 s

async def example_usage(port):
    url = f"http://localhost:${port}/metrics"
    metrics_poller = PollMetrics(metrics_url=url, interval=5.0)

    metric_snapshots = []
    # start consuming metrics in background
    async def consume():
        async for snap in metrics_poller.poll():
            print(f"[poll @ {snap['timestamp']:.1f}]")
            metric_snapshots.append(snap)

    metric_tasks = asyncio.create_task(consume())

    # run benchmarking workload
    await run_fake_benchmark()

    # stop polling
    metric_tasks.cancel()
    await asyncio.gather(metric_tasks, return_exceptions=True)

    with open("vllm_metrics_snapshots.json", "w") as f:
        json.dump(metric_snapshots, f, indent=2)

    print(f"Saved {len(metric_snapshots)} metric snapshots to vllm_metrics_snapshots.json")

if __name__ == "__main__":
    asyncio.run(example_usage(port=8066))