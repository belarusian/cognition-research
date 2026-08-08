#!/usr/bin/env python3
"""
context-benchmark.py

Benchmark practical Ollama context and chunk sizes for summarization.

Unlike a simple context-window test, this benchmark separates:

    1. Ollama runtime context size (num_ctx)
    2. Source-text chunk size
    3. Prompt evaluation speed
    4. Generation speed
    5. Model loading / context-allocation overhead
    6. End-to-end elapsed time

The purpose is to determine the best production settings for a bulk
summarization pipeline rather than merely testing the model's advertised
maximum context length.

Usage
-----

    ./context-benchmark.py book.txt

Specify another model:

    ./context-benchmark.py book.txt granite4.1:8b

Quick test:

    ./context-benchmark.py book.txt --quick

Test a specific runtime context:

    ./context-benchmark.py book.txt --contexts 32768

Test specific source sizes:

    ./context-benchmark.py book.txt \
        --contexts 32768 \
        --chunks 8000 12000 16000 20000 24000 28000

Dependencies
------------

Python 3 only. No third-party Python packages are required.

Ollama must be installed and running locally.

Output
------

Results are printed to the terminal and written to:

    context-benchmark.csv

Important
---------

The requested source-token size is necessarily approximate because this
script does not duplicate Granite's tokenizer. It adaptively estimates the
character/token ratio from Ollama's own prompt_eval_count results.

The ACTUAL PROMPT column is authoritative.
"""

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


OLLAMA_URL = "http://localhost:11434/api/generate"

DEFAULT_MODEL = "granite4.1:3b"

DEFAULT_CONTEXTS = [
    8192,
    16384,
    32768,
]

DEFAULT_CHUNKS = [
    4000,
    8000,
    12000,
    16000,
    20000,
    24000,
    28000,
]

DEFAULT_REPEATS = 2

NUM_PREDICT = 256

INITIAL_CHARS_PER_TOKEN = 4.0

# Leave this much room between the requested source chunk and num_ctx.
# This protects the instruction prompt and generated response.
CONTEXT_RESERVE = 1024

CSV_FILE = "context-benchmark.csv"


PROMPT_PREFIX = """Summarize the following text.

Preserve the important arguments, facts, concepts, names, and relationships.
Do not discuss the summarization process.
Return a concise but substantive summary.

TEXT:

"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama summarization context and chunk sizes."
    )

    parser.add_argument(
        "file",
        help="Large UTF-8 text file used as benchmark material.",
    )

    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=DEFAULT_CONTEXTS,
        help="Runtime context sizes to test.",
    )

    parser.add_argument(
        "--chunks",
        nargs="+",
        type=int,
        default=DEFAULT_CHUNKS,
        help="Approximate source-token sizes to test.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of measured runs per configuration.",
    )

    parser.add_argument(
        "--predict",
        type=int,
        default=NUM_PREDICT,
        help="Maximum generated summary tokens.",
    )

    parser.add_argument(
        "--csv",
        default=CSV_FILE,
        help=f"CSV output file (default: {CSV_FILE})",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short benchmark using only 32K context.",
    )

    return parser.parse_args()


def load_text(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def ns_to_seconds(value):
    if not value:
        return 0.0

    return value / 1_000_000_000


def token_rate(tokens, duration_ns):
    seconds = ns_to_seconds(duration_ns)

    if not seconds:
        return 0.0

    return tokens / seconds


def ollama_generate(
    model,
    source,
    context_size,
    num_predict,
):
    prompt = PROMPT_PREFIX + source

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_ctx": context_size,
            "num_predict": num_predict,
            "temperature": 0.0,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=1800,
    ) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    elapsed = time.perf_counter() - start

    return result, elapsed


def make_source(
    text,
    target_tokens,
    chars_per_token,
):
    target_chars = int(
        target_tokens * chars_per_token
    )

    target_chars = min(
        target_chars,
        len(text),
    )

    return text[:target_chars]


def calibrate(
    text,
    model,
    context_size,
    num_predict,
):
    """
    Estimate this corpus/model's characters-per-token ratio using Ollama's
    own prompt token count.

    The calibration source is intentionally moderate so that it does not
    approach the context limit.
    """

    print()
    print("Calibrating character/token ratio...")

    sample_chars = min(
        20000,
        len(text),
    )

    sample = text[:sample_chars]

    result, elapsed = ollama_generate(
        model,
        sample,
        context_size,
        min(num_predict, 32),
    )

    prompt_tokens = result.get(
        "prompt_eval_count",
        0,
    )

    if prompt_tokens <= 0:
        print(
            "[!] Ollama did not report prompt tokens; "
            "using fallback estimate."
        )
        return INITIAL_CHARS_PER_TOKEN

    # Subtract an approximate allowance for the fixed prompt.
    # The effect becomes negligible for production-sized chunks.
    chars_per_token = sample_chars / prompt_tokens

    print(
        f"    Sample characters: {sample_chars:,}"
    )
    print(
        f"    Prompt tokens:     {prompt_tokens:,}"
    )
    print(
        f"    Estimated chars/token: "
        f"{chars_per_token:.3f}"
    )
    print(
        f"    Calibration time:  {elapsed:.1f}s"
    )

    return chars_per_token


def extract_metrics(
    result,
    elapsed,
):
    prompt_tokens = result.get(
        "prompt_eval_count",
        0,
    )

    output_tokens = result.get(
        "eval_count",
        0,
    )

    prompt_duration = result.get(
        "prompt_eval_duration",
        0,
    )

    generation_duration = result.get(
        "eval_duration",
        0,
    )

    load_duration = result.get(
        "load_duration",
        0,
    )

    total_duration = result.get(
        "total_duration",
        0,
    )

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,

        "prompt_seconds":
            ns_to_seconds(prompt_duration),

        "generation_seconds":
            ns_to_seconds(generation_duration),

        "load_seconds":
            ns_to_seconds(load_duration),

        "ollama_total_seconds":
            ns_to_seconds(total_duration),

        "elapsed_seconds":
            elapsed,

        "prompt_tokens_per_second":
            token_rate(
                prompt_tokens,
                prompt_duration,
            ),

        "generation_tokens_per_second":
            token_rate(
                output_tokens,
                generation_duration,
            ),
    }


def median(values):
    if not values:
        return 0.0

    return statistics.median(values)


def summarize_runs(runs):
    return {
        "actual_prompt_tokens":
            round(
                median([
                    r["prompt_tokens"]
                    for r in runs
                ])
            ),

        "output_tokens":
            round(
                median([
                    r["output_tokens"]
                    for r in runs
                ])
            ),

        "prompt_tps":
            median([
                r["prompt_tokens_per_second"]
                for r in runs
            ]),

        "generation_tps":
            median([
                r["generation_tokens_per_second"]
                for r in runs
            ]),

        "prompt_seconds":
            median([
                r["prompt_seconds"]
                for r in runs
            ]),

        "generation_seconds":
            median([
                r["generation_seconds"]
                for r in runs
            ]),

        "load_seconds":
            median([
                r["load_seconds"]
                for r in runs
            ]),

        "elapsed_seconds":
            median([
                r["elapsed_seconds"]
                for r in runs
            ]),
    }


def classify_result(
    requested_chunk,
    actual_prompt,
    context_size,
):
    if actual_prompt >= context_size - 16:
        return "TRUNCATED"

    ratio = (
        actual_prompt / requested_chunk
        if requested_chunk
        else 0
    )

    if ratio < 0.75:
        return "UNDERSIZE"

    if ratio > 1.30:
        return "OVERSIZE"

    return "OK"


def write_csv(path, rows):
    fields = [
        "model",
        "context",
        "requested_chunk",
        "actual_prompt",
        "output_tokens",
        "prompt_tps",
        "generation_tps",
        "prompt_seconds",
        "generation_seconds",
        "load_seconds",
        "elapsed_seconds",
        "status",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_header():
    print()
    print(
        f"{'CTX':>8} "
        f"{'TARGET':>8} "
        f"{'ACTUAL':>8} "
        f"{'OUT':>6} "
        f"{'PROMPT/s':>10} "
        f"{'GEN/s':>9} "
        f"{'P.TIME':>8} "
        f"{'G.TIME':>8} "
        f"{'TOTAL':>8} "
        f"{'STATUS':>10}"
    )

    print("-" * 102)


def print_result(
    context,
    target,
    summary,
    status,
):
    print(
        f"{context:>8,} "
        f"{target:>8,} "
        f"{summary['actual_prompt_tokens']:>8,} "
        f"{summary['output_tokens']:>6,} "
        f"{summary['prompt_tps']:>10.1f} "
        f"{summary['generation_tps']:>9.1f} "
        f"{summary['prompt_seconds']:>7.1f}s "
        f"{summary['generation_seconds']:>7.1f}s "
        f"{summary['elapsed_seconds']:>7.1f}s "
        f"{status:>10}"
    )


def main():
    args = parse_args()

    text = load_text(args.file)

    contexts = sorted(
        set(args.contexts)
    )

    chunks = sorted(
        set(args.chunks)
    )

    if args.quick:
        contexts = [32768]
        chunks = [
            8000,
            12000,
            16000,
            20000,
            24000,
            28000,
        ]

    print()
    print("Ollama Summarization Benchmark")
    print("=" * 72)

    print(f"Model:       {args.model}")
    print(f"File:        {args.file}")
    print(f"Characters:  {len(text):,}")
    print(f"Repeats:     {args.repeats}")
    print(f"Output cap:  {args.predict:,} tokens")

    calibration_context = max(
        8192,
        min(contexts),
    )

    chars_per_token = calibrate(
        text,
        args.model,
        calibration_context,
        args.predict,
    )

    rows = []

    print_header()

    for context_size in contexts:

        for requested_chunk in chunks:

            # Do not intentionally construct a source prompt that consumes
            # essentially the entire context window.
            if (
                requested_chunk
                + args.predict
                + CONTEXT_RESERVE
                >= context_size
            ):
                continue

            source = make_source(
                text,
                requested_chunk,
                chars_per_token,
            )

            runs = []

            failed = None

            for repeat in range(
                1,
                args.repeats + 1,
            ):
                try:
                    result, elapsed = ollama_generate(
                        args.model,
                        source,
                        context_size,
                        args.predict,
                    )

                    runs.append(
                        extract_metrics(
                            result,
                            elapsed,
                        )
                    )

                except urllib.error.HTTPError as e:
                    failed = f"HTTP-{e.code}"
                    break

                except urllib.error.URLError as e:
                    failed = f"URL-ERROR"
                    break

                except TimeoutError:
                    failed = "TIMEOUT"
                    break

                except Exception as e:
                    failed = (
                        type(e).__name__
                    )
                    break

            if failed:
                print(
                    f"{context_size:>8,} "
                    f"{requested_chunk:>8,} "
                    f"{'-':>8} "
                    f"{'-':>6} "
                    f"{'-':>10} "
                    f"{'-':>9} "
                    f"{'-':>8} "
                    f"{'-':>8} "
                    f"{'-':>8} "
                    f"{failed:>10}"
                )

                continue

            summary = summarize_runs(runs)

            status = classify_result(
                requested_chunk,
                summary[
                    "actual_prompt_tokens"
                ],
                context_size,
            )

            print_result(
                context_size,
                requested_chunk,
                summary,
                status,
            )

            rows.append({
                "model":
                    args.model,

                "context":
                    context_size,

                "requested_chunk":
                    requested_chunk,

                "actual_prompt":
                    summary[
                        "actual_prompt_tokens"
                    ],

                "output_tokens":
                    summary[
                        "output_tokens"
                    ],

                "prompt_tps":
                    round(
                        summary["prompt_tps"],
                        2,
                    ),

                "generation_tps":
                    round(
                        summary[
                            "generation_tps"
                        ],
                        2,
                    ),

                "prompt_seconds":
                    round(
                        summary[
                            "prompt_seconds"
                        ],
                        3,
                    ),

                "generation_seconds":
                    round(
                        summary[
                            "generation_seconds"
                        ],
                        3,
                    ),

                "load_seconds":
                    round(
                        summary[
                            "load_seconds"
                        ],
                        3,
                    ),

                "elapsed_seconds":
                    round(
                        summary[
                            "elapsed_seconds"
                        ],
                        3,
                    ),

                "status":
                    status,
            })

            # Save after every configuration so interrupted long benchmarks
            # still leave useful results.
            write_csv(
                args.csv,
                rows,
            )

    print()
    print("=" * 72)

    if rows:
        write_csv(
            args.csv,
            rows,
        )

        print(
            f"Results saved to: {args.csv}"
        )

        usable = [
            r
            for r in rows
            if r["status"] == "OK"
        ]

        if usable:
            fastest = min(
                usable,
                key=lambda r:
                    r["elapsed_seconds"],
            )

            largest_fast = max(
                usable,
                key=lambda r: (
                    r["actual_prompt"],
                    r["prompt_tps"],
                ),
            )

            print()
            print("Quick observations")
            print("-" * 72)

            print(
                "Fastest measured configuration:"
            )

            print(
                f"    context={fastest['context']:,}, "
                f"chunk≈{fastest['actual_prompt']:,} tokens, "
                f"elapsed={fastest['elapsed_seconds']:.1f}s"
            )

            print()
            print(
                "Largest successful prompt:"
            )

            print(
                f"    context={largest_fast['context']:,}, "
                f"prompt={largest_fast['actual_prompt']:,} tokens, "
                f"prompt={largest_fast['prompt_tps']:.1f} tok/s, "
                f"generation={largest_fast['generation_tps']:.1f} tok/s"
            )

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("[!] Benchmark interrupted.")
        sys.exit(130)

    except Exception as e:
        print(
            f"[!] Fatal error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

