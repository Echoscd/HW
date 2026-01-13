# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""

import argparse
import asyncio
import gc
import json
import os
import random
import time
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Optional
from loguru import logger

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)

from poll_metrics import PollMetrics

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_dataset import (
    AIMODataset,
    ASRDataset,
    ArxivSummarizationDataset,
    BurstGPTDataset,
    ConversationDataset,
    CustomDataset_from_previous_Benchmark,
    CustomDataset,
    HuggingFaceDataset,
    InstructCoderDataset,
    MTBenchDataset,
    NextEditPredictionDataset,
    RandomDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    VisionArenaDataset,
)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


def _get_current_request_rate(
    ramp_up_strategy: Optional[Literal["linear", "exponential"]],
    ramp_up_start_rps: Optional[int],
    ramp_up_end_rps: Optional[int],
    request_index: int,
    total_requests: int,
    request_rate: float,
) -> float:
    if (
        ramp_up_strategy
        and ramp_up_start_rps is not None
        and ramp_up_end_rps is not None
    ):
        progress = request_index / max(total_requests - 1, 1)
        if ramp_up_strategy == "linear":
            increase = (ramp_up_end_rps - ramp_up_start_rps) * progress
            return ramp_up_start_rps + increase
        elif ramp_up_strategy == "exponential":
            ratio = ramp_up_end_rps / ramp_up_start_rps
            return ramp_up_start_rps * (ratio**progress)
        else:
            raise ValueError(f"Unknown ramp-up strategy: {ramp_up_strategy}")
    return request_rate


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
    ramp_up_strategy: Optional[Literal["linear", "exponential"]] = None,
    ramp_up_start_rps: Optional[int] = None,
    ramp_up_end_rps: Optional[int] = None,
) -> AsyncGenerator[tuple[SampleRequest, float], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness and OPTIONAL ramp-up strategy.

    Args:
        input_requests:
            A list of input requests, each represented as a SampleRequest.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
         ramp_up_strategy (optional):
            The ramp-up strategy. Can be "linear" or "exponential".
            If None, uses constant request rate (specified by request_rate).
        ramp_up_start_rps (optional):
            The starting request rate for ramp-up.
        ramp_up_end_rps (optional):
            The ending request rate for ramp-up.
    """
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    # Convert to list to get length for ramp-up calculations
    if isinstance(input_requests, Iterable) and not isinstance(input_requests, list):
        input_requests = list(input_requests)

    total_requests = len(input_requests)
    request_index = 0

    for request in input_requests:
        current_request_rate = _get_current_request_rate(
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
            request_index,
            total_requests,
            request_rate,
        )

        yield request, current_request_rate

        request_index += 1

        if current_request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        theta = 1.0 / (current_request_rate * burstiness)

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


async def benchmark(
    args,
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: Optional[int],
    lora_modules: Optional[Iterable[str]],
    extra_body: Optional[dict],
    ramp_up_strategy: Optional[Literal["linear", "exponential"]] = None,
    ramp_up_start_rps: Optional[int] = None,
    ramp_up_end_rps: Optional[int] = None,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    logger.info("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
        input_requests[0].expected_output_len,
        input_requests[0].multi_modal_data,
    )

    assert test_mm_content is None or isinstance(test_mm_content, dict)
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
        extra_body=extra_body,
    )

    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        logger.info("Initial test run completed. Starting main benchmark run...")

    if lora_modules:
        # For each input request, choose a LoRA module at random.
        lora_modules = iter(
            [random.choice(lora_modules) for _ in range(len(input_requests))]
        )

    if profile:
        logger.info("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            logger.info("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    if ramp_up_strategy is not None:
        logger.info(
            f"Traffic ramp-up strategy: {ramp_up_strategy}. Will increase "
            f"RPS from {ramp_up_start_rps} to {ramp_up_end_rps} RPS over "
            "the duration of the benchmark."
        )
    else:
        logger.info(f"Traffic request rate: {request_rate} RPS.")

    logger.info(f"Burstiness factor: {burstiness} ({distribution})")
    logger.info(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.time()
    tasks: list[asyncio.Task] = []

    rps_change_events = []
    last_int_rps = -1
    if ramp_up_strategy is not None and ramp_up_start_rps is not None:
        last_int_rps = ramp_up_start_rps
        rps_change_events.append(
            {
                "rps": last_int_rps,
                "timestamp": datetime.now().isoformat(),
            }
        )

    # 准备输出文件
    output_file = None
    if hasattr(args, 'save_result') and args.save_result:
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}"
            if args.max_concurrency is not None
            else ""
        )
        if args.ramp_up_strategy is not None:
            file_name = f"{backend}-ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-{current_dt}-detailed.jsonl"
        else:
            file_name = f"{backend}-{request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}-detailed.jsonl"
        if hasattr(args, 'result_filename') and args.result_filename:
            file_name = args.result_filename.replace('.json', '.jsonl')
        if hasattr(args, 'result_dir') and args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
        output_file = open(file_name, 'w', encoding='utf-8')
        logger.info(f"Writing detailed results to {file_name}")

    async for request, current_request_rate in get_request(
        input_requests,
        request_rate,
        burstiness,
        ramp_up_strategy,
        ramp_up_start_rps,
        ramp_up_end_rps,
    ):
        if ramp_up_strategy is not None:
            current_int_rps = int(current_request_rate)
            if current_int_rps > last_int_rps:
                timestamp = datetime.now().isoformat()
                for rps_val in range(last_int_rps + 1, current_int_rps + 1):
                    rps_change_events.append({"rps": rps_val, "timestamp": timestamp})
                last_int_rps = current_int_rps

        prompt, prompt_len, output_len, mm_content = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multi_modal_data,
        )
        req_model_id, req_model_name = model_id, model_name
        if lora_modules:
            req_lora_module = next(lora_modules)
            req_model_id, req_model_name = req_lora_module, req_lora_module

        if args.pass_output_len_to_http_request:
            # pass in num_output_tokens via the extrabody
            extra_body["vllm_xargs"] = {
                "num_output_tokens": output_len
            }
        if args.pass_output_len_as_priority:
            extra_body["priority"] = output_len

        request_func_input = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        
        # 创建包装函数，在任务完成时立即写入文件
        async def wrapped_request_func():
            output = await limited_request_func(request_func_input=request_func_input, pbar=pbar)
            
            # 立即写入文件
            if output_file is not None:
                # 计算输出长度
                output_len = output.output_tokens
                if not output_len and output.success:
                    # 如果没有output_tokens，使用tokenizer计算
                    output_len = len(
                        tokenizer(
                            output.generated_text, add_special_tokens=False
                        ).input_ids
                    )
                
                result_record = {
                    "prompt_length": output.prompt_len,
                    "prompt_content": output.prompt,
                    "inference_time": output.latency,  # 推理时间（秒）
                    "output_content": output.generated_text,
                    "output_length": output_len,
                    "success": output.success,
                    "error": output.error if not output.success else None,
                }
                output_file.write(json.dumps(result_record, ensure_ascii=False) + '\n')
                output_file.flush()  # 立即刷新到磁盘
            
            return output
        
        task = asyncio.create_task(wrapped_request_func())
        tasks.append(task)
    
    # 等待所有任务完成
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    
    if output_file is not None:
        output_file.close()
        logger.info(f"Finished writing detailed results")

    if profile:
        logger.info("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            logger.info("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_end_time = time.time()
    benchmark_duration = benchmark_end_time - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    logger.info("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    logger.info("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    logger.info("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    logger.info("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    logger.info("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    logger.info(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    if goodput_config_dict:
        logger.info(
            "{:<40} {:<10.2f}".format(
                "Request goodput (req/s):", metrics.request_goodput
            )
        )
    logger.info(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    logger.info(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    result = {
        "start_time": benchmark_start_time,
        "end_time": benchmark_end_time,
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "prompts": [output.prompt for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    if rps_change_events:
        result["rps_change_events"] = rps_change_events

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    # 统计 prompt 和输出长度
    logger.info("{s:{c}^{n}}".format(s=" Length Statistics ", n=50, c="-"))
    
    # 计算 prompt 长度统计
    prompt_lens = [output.prompt_len for output in outputs if output.success]
    if prompt_lens:
        prompt_percentiles = [10, 25, 50, 75, 90, 99]
        logger.info("{:<40} {:<10.2f}".format("Mean Prompt Length (tokens):", np.mean(prompt_lens)))
        for p in prompt_percentiles:
            p_value = np.percentile(prompt_lens, p)
            logger.info("{:<40} {:<10.2f}".format(f"P{p} Prompt Length (tokens):", p_value))
            result[f"p{p}_prompt_length"] = float(p_value)
        result["mean_prompt_length"] = float(np.mean(prompt_lens))
    
    # 计算输出长度统计
    if actual_output_lens:
        output_lens = [len for len in actual_output_lens if len > 0]
        if output_lens:
            output_percentiles = [10, 25, 50, 75, 90, 99]
            logger.info("{:<40} {:<10.2f}".format("Mean Output Length (tokens):", np.mean(output_lens)))
            for p in output_percentiles:
                p_value = np.percentile(output_lens, p)
                logger.info("{:<40} {:<10.2f}".format(f"P{p} Output Length (tokens):", p_value))
                result[f"p{p}_output_length"] = float(p_value)
            result["mean_output_length"] = float(np.mean(output_lens))

    logger.info("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                )
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any], file_name: str
) -> None:
    metrics = [
        "median_ttft_ms",
        "mean_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]] for k in metrics},
        extra_info={
            k: results[k]
            for k in results
            if k not in metrics and k not in ignored_metrics
        },
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)

def _validate_ramp_up_strategy(args, request_rate):
    if request_rate != float("inf"):
        raise ValueError(
            "When using ramp-up, do not specify --request-rate. "
            "The request rate will be controlled by ramp-up parameters. "
            "Please remove the --request-rate argument."
        )
    if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
        raise ValueError(
            "When using --ramp-up-strategy, both --ramp-up-start-rps and "
            "--ramp-up-end-rps must be specified"
        )
    if args.ramp_up_start_rps < 0 or args.ramp_up_end_rps < 0:
        raise ValueError("Ramp-up start and end RPS must be non-negative")
    if args.ramp_up_start_rps > args.ramp_up_end_rps:
        raise ValueError("Ramp-up start RPS must be less than end RPS")
    if args.ramp_up_strategy == "exponential" and args.ramp_up_start_rps == 0:
        raise ValueError("For exponential ramp-up, the start RPS cannot be 0.")

def _get_api_base_url(args):
    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"
    return api_url, base_url

def _get_input_requests(args, tokenizer):
    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
        )

    if args.dataset_name == "custom_from_previous_benchmark":
        logger.info(f"Using {args.dataset_name}!!!")
        logger.info(f"Thus using output lengths from previous benchmark results")
        dataset = CustomDataset_from_previous_Benchmark(dataset_path=args.dataset_path)
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            skip_chat_template=args.custom_skip_chat_template,
        )

    elif args.dataset_name == "custom":
        dataset = CustomDataset(dataset_path=args.dataset_path)
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.custom_output_len,
            skip_chat_template=args.custom_skip_chat_template,
        )

    elif args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.backend == "openai-chat":
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=False,
            )
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset."
            )
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=True,
            )

    elif args.dataset_name == "hf":
        # all following datasets are implemented from the
        # HuggingFaceDataset base class
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = VisionArenaDataset
            args.hf_split = "train"
            args.hf_subset = None
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ConversationDataset
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_class = AIMODataset
            args.hf_split = "train"
        elif args.dataset_path in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS:  # noqa: E501
            dataset_class = NextEditPredictionDataset
            args.hf_split = "train"
        elif args.dataset_path in ASRDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ASRDataset
            args.hf_split = "train"
        else:
            supported_datasets = set(
                [
                    dataset_name
                    for cls in HuggingFaceDataset.__subclasses__()
                    for dataset_name in cls.SUPPORTED_DATASET_PATHS
                ]
            )
            raise ValueError(
                f"Unsupported dataset path: {args.dataset_path}. "
                "Huggingface dataset only supports dataset_path"
                f" from one of following: {supported_datasets}. "
                "Please consider contributing if you would "
                "like to add support for additional dataset formats."
            )

        if dataset_class.IS_MULTIMODAL and backend not in [
            "openai-chat",
            "openai-audio",
        ]:
            # multi-modal benchmark is only available on OpenAI Chat backend.
            raise ValueError(
                "Multi-modal content is only supported on 'openai-chat' and "
                "'openai-audio' backend."
            )
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
        )

    elif args.dataset_name == "arxiv-summarization":
        dataset_path = args.dataset_path or "ccdv/arxiv-summarization"
        dataset_split = args.hf_split or "train"
        dataset = ArxivSummarizationDataset(
            dataset_path=dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=dataset_split,
            random_seed=args.seed,
        )
        input_requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            output_len=args.hf_output_len,
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        dataset_mapping = {
            "sharegpt": lambda: ShareGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                output_len=args.sharegpt_output_len,
            ),
            "burstgpt": lambda: BurstGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(tokenizer=tokenizer, num_requests=args.num_prompts),
            "random": lambda: RandomDataset(dataset_path=args.dataset_path).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
            ),
        }

        try:
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err

    return input_requests

async def main(args: argparse.Namespace):
    logger.info(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode


    # Loop through all request rate, treat each one like a new benchmark.
    for request_rate in args.request_rate:

        # ====================== Benchmark config setup ==========================================
        logger.info(f"==== RUNNING BENCHMARK FOR REQUEST-RATE {request_rate} OUT OF {args.request_rate} RRs =====")

        # Validate ramp-up arguments
        if args.ramp_up_strategy is not None: _validate_ramp_up_strategy(args, request_rate)

        # Get api_url and base_url
        api_url, base_url = _get_api_base_url(args)

        # Get tokenizer
        tokenizer = get_tokenizer(tokenizer_id, tokenizer_mode=tokenizer_mode, trust_remote_code=args.trust_remote_code)

        # Get dataset/input_requests
        input_requests = _get_input_requests(args, tokenizer)
        
        # get goodput stuff
        goodput_config_dict = check_goodput_args(args)

        # Collect the sampling parameters.
        sampling_params = {
            k: v
            for k, v in {
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "temperature": args.temperature,
                # "vllm_xargs": { "num_output_tokens": 100 },
            }.items()
            if v is not None
        }

        # Sampling parameters are only supported by openai-compatible backend.
        if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
            raise ValueError("Sampling parameters are only supported by openai-compatible backends.")

        # if no temp provided, use temp=0.0
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 0.0  

        # disable prompt caching in llama.cpp backend
        if args.backend == "llama.cpp":
            sampling_params["cache_prompt"] = False

        # set custom max completion tokens
        if args.set_max_completion_tokens:
            assert args.backend == "openai-chat", f"Backend must be openai-chat"
            sampling_params["max_completion_tokens"] = args.set_max_completion_tokens

        # Avoid GC processing "static" data - reduce pause times.
        gc.collect()
        gc.freeze()

        # ================================================================================================

        logger.info(f"Benchmark starting soon...")
        """
        Begin metrics polling
        """
        metrics_url = f"{base_url}/metrics"
        metrics_poll_interval = 1.0
        logger.info(f"Metrics polling every {metrics_poll_interval}s to {metrics_url} will begin now")
        metrics_poller = PollMetrics(metrics_url=metrics_url, interval=metrics_poll_interval)
        
        # start consuming metrics in background
        metric_snapshots = []
        async def consume():
            async for snap in metrics_poller.poll():
                metric_snapshots.append(snap)
        metric_tasks = asyncio.create_task(consume())


        """
        Begin benchmark
        """
        logger.info(f"Starting benchmark...")

        benchmark_result = await benchmark(
            args=args,
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            request_rate=request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
            lora_modules=args.lora_modules,
            extra_body=sampling_params,
            ramp_up_strategy=args.ramp_up_strategy,
            ramp_up_start_rps=args.ramp_up_start_rps,
            ramp_up_end_rps=args.ramp_up_end_rps,
        )

        # 3) Stop poller
        """
        Stop metrics polling
        """
        metric_tasks.cancel()
        await asyncio.gather(metric_tasks, return_exceptions=True)

        """
        Save result after benchmark
        """
        # Save config and results to json
        if args.save_result or args.append_result:
            result_json: dict[str, Any] = {}

            # Setup
            current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_json["date"] = current_dt
            result_json["backend"] = backend
            result_json["model_id"] = model_id
            result_json["dataset_name"] = args.dataset_name
            result_json["dataset_path"] = args.dataset_path
            result_json["tokenizer_id"] = tokenizer_id
            result_json["num_prompts"] = args.num_prompts

            # Metadata
            if args.metadata:
                for item in args.metadata:
                    if "=" in item:
                        kvstring = item.split("=")
                        result_json[kvstring[0].strip()] = kvstring[1].strip()
                    else:
                        raise ValueError(
                            "Invalid metadata format. Please use KEY=VALUE format."
                        )
            # Traffic
            result_json["request_rate"] = (
                request_rate if request_rate < float("inf") else "inf"
            )
            result_json["burstiness"] = args.burstiness
            result_json["max_concurrency"] = args.max_concurrency

            if args.ramp_up_strategy is not None:
                result_json["ramp_up_strategy"] = args.ramp_up_strategy
                result_json["ramp_up_start_rps"] = args.ramp_up_start_rps
                result_json["ramp_up_end_rps"] = args.ramp_up_end_rps

            # Merge with benchmark result
            result_json = {**result_json, **benchmark_result}

            # Add metric snapshots in as well!
            result_json["metrics_snapshots"] = metric_snapshots

            if not args.save_detailed:
                # Remove fields with too many data points
                FIELDS_TO_REMOVE = [
                    # "input_lens", # we wanna keep input lengths
                    # "output_lens", # we wanna keep output lens
                    "ttfts",
                    "itls",
                    "metrics_snapshots", # we dont need metric snapshots if not detailed!
                    # "generated_texts", # keep generated texts
                    # "errors", # keep errors
                ]
                logger.info(f"Since we don't saved detailed, we remove {FIELDS_TO_REMOVE}")
                for field in FIELDS_TO_REMOVE:
                    if field in result_json:
                        del result_json[field]
                    if field in benchmark_result:
                        del benchmark_result[field]

            # Save to file
            base_model_id = model_id.split("/")[-1]
            max_concurrency_str = (
                f"-concurrency{args.max_concurrency}"
                if args.max_concurrency is not None
                else ""
            )
            if args.ramp_up_strategy is not None:
                file_name = f"{backend}-ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
            else:
                file_name = f"{backend}-{request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
            if args.result_filename:
                file_name = args.result_filename
            if args.result_dir:
                os.makedirs(args.result_dir, exist_ok=True)
                file_name = os.path.join(args.result_dir, file_name)
            with open(
                file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
            ) as outfile:
                # Append a newline.
                if args.append_result and outfile.tell() != 0:
                    outfile.write("\n")
                json.dump(result_json, outfile)
            save_to_pytorch_benchmark_format(args, result_json, file_name)


def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=[
            "sharegpt",
            "burstgpt",
            "sonnet",
            "random",
            "hf",
            "custom",
            "custom_from_previous_benchmark",
            "arxiv-summarization",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the sharegpt/sonnet dataset. "
        "Or the huggingface dataset ID if using HF dataset.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=(
            "Number of logprobs-per-token to compute & return as part of "
            "the request. If unspecified, then either (1) if beam search "
            "is disabled, no logprobs are computed & a single dummy "
            "logprob is returned for each token; or (2) if beam search "
            "is enabled 1 logprob per token is computed"
        ),
    )
    parser.add_argument(
        "--request-rate",
        nargs='+',
        type=float,
        default=[float("inf")], # [1, 2, 4, 10]
        help="Number of RPS, if this is inf, then"
        "all requests are sent at time 0"
        "Pass as list of integers!"
    )
    # parser.add_argument(
    #     "--request-rate",
    #     type=float,
    #     default=float("inf"),
    #     help="Number of requests per second. If this is inf, "
    #     "then all the requests are sent at time 0. "
    #     "Otherwise, we use Poisson process or gamma distribution "
    #     "to synthesize the request arrival times.",
    # )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="When saving the results, whether to include per request "
        "information such as response, error, ttfs, tpots, etc.",
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="Append the benchmark result to the existing json file.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-separated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'Default value is "ttft,tpot,itl".',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="1,25,50,75,95,99",
        help="Comma-separated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99". '
        'Use "--percentile-metrics" to select metrics.',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='Specify service level objectives for goodput as "KEY:VALUE" '
        "pairs, where the key is a metric name, and the value is in "
        'milliseconds. Multiple "KEY:VALUE" pairs can be provided, '
        "separated by spaces. Allowed request level metric names are "
        '"ttft", "tpot", "e2el". For more context on the definition of '
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
    )
    parser.add_argument(
        "--pass-output-len-to-http-request",
        action="store_true",
        help="Pass in the request.output_len to the sampling_params xargs"
        'This is a valid option when vllm is served with the custom EfficientScheduler'
        'where it can consider output length!!!'
    )
    parser.add_argument(
        "--set-max-completion-tokens",
        type=int,
        default=None,
        help="Pass in a custom max completions tokens parameter"
    )
    parser.add_argument(
        "--pass-output-len-as-priority",
        action="store_true",
        help="Pass in a output length as priority parameter to be used when scheduler.policy=sjf in serve"
    )

    # group for dataset specific arguments
    custom_group = parser.add_argument_group("custom dataset options")
    custom_group.add_argument(
        "--custom-output-len",
        type=int,
        default=256,
        help="Number of output tokens per request, used only for custom dataset.",
    )
    custom_group.add_argument(
        "--custom-skip-chat-template",
        action="store_true",
        help="Skip applying chat template to prompt, used only for custom dataset.",
    )

    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help="Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help="Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=(
            "Number of fixed prefix tokens before the random context "
            "in a request. "
            "The total input length is the sum of `random-prefix-len` and "
            "a random "
            "context length sampled from [input_len * (1 - range_ratio), "
            "input_len * (1 + range_ratio)]."
        ),
    )

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument(
        "--hf-subset", type=str, default=None, help="Subset of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-split", type=str, default=None, help="Split of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling parameter. Only has effect on openai-compatible backends.",
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature sampling parameter. Only has effect on "
        "openai-compatible backends. If not specified, default to greedy "
        "decoding (i.e. temperature==0.0).",
    )

    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "slow", "mistral", "custom"],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        "always use the slow tokenizer. \n* "
        '"mistral" will always use the `mistral_common` tokenizer. \n*'
        '"custom" will use --tokenizer to select the preregistered tokenizer.',
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the ``--model`` argument. ",
    )

    parser.add_argument(
        "--lora-modules",
        nargs="+",
        default=None,
        help="A subset of LoRA module names passed in when "
        "launching the server. For each request, the "
        "script chooses a LoRA module at random.",
    )

    parser.add_argument(
        "--ramp-up-strategy",
        type=str,
        default=None,
        choices=["linear", "exponential"],
        help="The ramp-up strategy. This would be used to "
        "ramp up the request rate from initial RPS to final "
        "RPS rate (specified by --ramp-up-start-rps and --ramp-up-end-rps). "
        "over the duration of the benchmark.",
    )
    parser.add_argument(
        "--ramp-up-start-rps",
        type=int,
        default=None,
        help="The starting request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )
    parser.add_argument(
        "--ramp-up-end-rps",
        type=int,
        default=None,
        help="The ending request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
    # main(args)
