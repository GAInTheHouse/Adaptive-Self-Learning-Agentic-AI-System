#!/usr/bin/env python3
"""
Unified evaluation: run any dataset on baseline and on improved (fine-tuned) models if they exist.
Replaces run_comprehensive_evaluations, kavya_evaluation_framework, run_benchmark, verify_evaluation_numbers.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline_model import BaselineSTTModel
from src.evaluation.metrics import EvaluationModule
from src.utils.model_versioning import get_all_model_versions, get_current_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_gold_from_llm(stt_transcript: str, llm_corrector) -> str:
    """Get LLM-corrected transcript as gold (same pattern as scripts/finetune_wav2vec2.py)."""
    if not stt_transcript:
        return ""
    if not llm_corrector or not llm_corrector.is_available():
        return stt_transcript
    llm_result = llm_corrector.correct_transcript(stt_transcript, errors=[], context={})
    gold = llm_result.get("corrected_transcript", stt_transcript).strip()
    gold = re.sub(r'^["\'](.*)["\']$', r"\1", gold.strip())
    return gold.strip()


def discover_audio_with_llm_gold(audio_dir: Path, llm_model_name: str = "llama3.2:3b") -> List[Tuple[str, str]]:
    """
    Discover WAV/MP3 in audio_dir and use LLM as gold standard.
    Requires Ollama running with the given model. Same LLM gold logic as scripts/finetune_wav2vec2.py.
    Returns list of (audio_path, gold_reference).
    """
    from src.agent.llm_corrector import LlamaLLMCorrector

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")
    files = sorted(
        list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")),
        key=lambda p: p.name,
    )
    paths = [str(f) for f in files]
    if not paths:
        logger.warning(f"No WAV/MP3 found in {audio_dir}")
        return []

    llm_corrector = LlamaLLMCorrector(model_name=llm_model_name, raise_on_error=False)
    if not llm_corrector.is_available():
        logger.error(
            "LLM is not available. Start Ollama (e.g. 'ollama serve') and pull the model "
            f"(e.g. 'ollama pull {llm_model_name}'). Then re-run with --gold-from-llm."
        )
        sys.exit(1)
    logger.info(f"Using LLM ({llm_model_name}) as gold standard for {len(paths)} files.")

    baseline = BaselineSTTModel(model_name="whisper")
    pairs = []
    for i, audio_path in enumerate(paths):
        try:
            stt_result = baseline.transcribe(audio_path)
            stt_transcript = (stt_result.get("transcript") or "").strip()
            gold = _get_gold_from_llm(stt_transcript, llm_corrector)
            pairs.append((audio_path, gold))
            if (i + 1) % 10 == 0:
                logger.info(f"  LLM gold: {i + 1}/{len(paths)} files")
        except Exception as e:
            logger.warning(f"Failed {audio_path}: {e}")
    return pairs


def load_eval_set(eval_set_path: Path) -> List[Tuple[str, str]]:
    """Load (audio_path, reference) pairs from JSON, JSONL, or CSV."""
    path = Path(eval_set_path)
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    pairs = []
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("samples", data.get("data", []))
        for item in items:
            ap = item.get("audio_path") or item.get("audio") or item.get("path")
            ref = item.get("reference") or item.get("text") or item.get("target_text") or ""
            if ap and ref is not None:
                pairs.append((str(ap).strip(), str(ref).strip()))
    elif suffix == ".jsonl":
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                ap = item.get("audio_path") or item.get("audio") or item.get("path")
                ref = item.get("reference") or item.get("text") or item.get("target_text") or ""
                if ap and ref is not None:
                    pairs.append((str(ap).strip(), str(ref).strip()))
    elif suffix == ".csv":
        import csv
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ap = row.get("audio_path") or row.get("audio") or row.get("path")
                ref = row.get("reference") or row.get("text") or row.get("target_text") or ""
                if ap and ref is not None:
                    pairs.append((str(ap).strip(), str(ref).strip()))
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .json, .jsonl, or .csv")
    return pairs


def discover_audio_and_refs(audio_dir: Path, refs_path: Optional[Path]) -> List[Tuple[str, str]]:
    """Discover *.wav and *.mp3 in audio_dir; pair with refs from refs_path if given."""
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")
    files = sorted(
        list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")),
        key=lambda p: p.name,
    )
    paths = [str(f) for f in files]
    if not refs_path or not refs_path.exists():
        logger.warning("No --refs file: cannot compute WER. Provide refs (JSON/JSONL/CSV with audio_path + reference) for metrics.")
        return [(p, "") for p in paths]
    pairs_from_refs = load_eval_set(refs_path)
    refs_by_path = {str(Path(p).resolve()): r for p, r in pairs_from_refs}
    refs_by_name = {Path(p).name: r for p, r in pairs_from_refs}
    out = []
    for p in paths:
        r = refs_by_path.get(p) or refs_by_path.get(str(Path(p).resolve())) or refs_by_name.get(Path(p).name, "")
        out.append((p, r))
    return out


def run_model_on_set(
    model: BaselineSTTModel,
    pairs: List[Tuple[str, str]],
    model_id: str,
    skip_empty_ref: bool = True,
    include_per_sample: bool = False,
) -> Dict[str, Any]:
    """Run model on all (audio_path, reference) pairs; return metrics (and optionally per-sample results)."""
    eval_mod = EvaluationModule()
    latencies = []
    for audio_path, reference in pairs:
        if skip_empty_ref and not reference.strip():
            continue
        try:
            start = time.time()
            out = model.transcribe(audio_path)
            latencies.append(time.time() - start)
            hyp = (out.get("transcript") or "").strip()
            eval_mod.add_prediction(reference, hyp)
        except Exception as e:
            logger.warning(f"{model_id} failed on {audio_path}: {e}")
    metrics = eval_mod.get_metrics()
    if not metrics:
        out = {"wer": None, "cer": None, "num_samples": 0, "latency_mean_sec": None}
    else:
        out = {
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "num_samples": metrics["num_samples"],
            "latency_mean_sec": sum(latencies) / len(latencies) if latencies else None,
        }
    if include_per_sample:
        out["results"] = eval_mod.results
    return out


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Evaluate a dataset on baseline and improved models (if any)."
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        help="Path to evaluation set: JSON/JSONL/CSV with audio_path (or audio) and reference (or text/target_text).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="Directory of WAV/MP3 files. Use with --refs to provide references for WER.",
    )
    parser.add_argument(
        "--refs",
        type=Path,
        help="Path to refs file (same format as --eval-set). Used only with --audio-dir (ignored if --gold-from-llm).",
    )
    parser.add_argument(
        "--gold-from-llm",
        action="store_true",
        help="Use LLM (Ollama) as gold standard. Requires --audio-dir. Checks Ollama is running and gets reference from LLM-corrected baseline transcript.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model name for --gold-from-llm (default: llama3.2:3b).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/evaluation_outputs"),
        help="Output directory for report and JSON.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline (Whisper), skip improved models.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Include latency/throughput benchmark in output.",
    )
    args = parser.parse_args()

    if args.gold_from_llm:
        if not args.audio_dir:
            logger.error("--gold-from-llm requires --audio-dir.")
            sys.exit(1)
        pairs = discover_audio_with_llm_gold(args.audio_dir, args.llm_model)
    elif args.eval_set:
        pairs = load_eval_set(args.eval_set)
    elif args.audio_dir:
        pairs = discover_audio_and_refs(args.audio_dir, args.refs)
    else:
        logger.error("Provide --eval-set, or --audio-dir (with --refs or --gold-from-llm).")
        sys.exit(1)

    pairs_with_ref = [(p, r) for p, r in pairs if r.strip()]
    if not pairs_with_ref:
        logger.error("No (audio_path, reference) pairs with non-empty reference. Cannot compute WER.")
        sys.exit(1)

    logger.info(f"Loaded {len(pairs_with_ref)} samples with references.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "num_samples": len(pairs_with_ref),
        "baseline": {},
        "improved_models": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gold_source": "llm" if args.gold_from_llm else ("eval_set" if args.eval_set else "refs"),
    }

    baseline = BaselineSTTModel(model_name="whisper")
    baseline_info = baseline.get_model_info()
    report["baseline"] = {
        "model_id": "whisper",
        "model_info": baseline_info,
        **run_model_on_set(baseline, pairs_with_ref, "baseline"),
    }
    logger.info(f"Baseline WER: {report['baseline']['wer']:.4f}, CER: {report['baseline']['cer']:.4f}")

    if not args.baseline_only:
        versions = get_all_model_versions()
        current_path = get_current_model_path()
        for v in versions:
            model_path = v["path"]
            version_num = v["version_num"]
            model_id = f"wav2vec2-finetuned-v{version_num}"
            try:
                model = BaselineSTTModel(model_name=model_id)
                metrics = run_model_on_set(model, pairs_with_ref, model_id)
                metrics["model_id"] = model_id
                metrics["path"] = model_path
                metrics["is_current"] = (model_path == current_path)
                report["improved_models"].append(metrics)
                logger.info(f"{model_id} WER: {metrics['wer']:.4f}, CER: {metrics['cer']:.4f}")
            except Exception as e:
                logger.warning(f"Could not load or run {model_id}: {e}")

    if args.benchmark:
        from src.benchmark import BaselineBenchmark
        audio_paths = [p for p, _ in pairs_with_ref[: min(50, len(pairs_with_ref))]]
        if audio_paths:
            bench = BaselineBenchmark(model_name="whisper")
            bench_report = bench.generate_report(audio_paths)
            report["benchmark"] = bench_report
            bench.save_report(bench_report, str(output_dir / "benchmark_report.json"))

    out_json = output_dir / "evaluation_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {out_json}")

    txt_lines = [
        "=" * 60,
        "EVALUATION REPORT",
        f"Samples: {report['num_samples']}",
        "",
        "Baseline (Whisper)",
        f"  WER: {report['baseline']['wer']:.4f}  CER: {report['baseline']['cer']:.4f}",
        "",
    ]
    for m in report["improved_models"]:
        txt_lines.append(f"{m['model_id']}  WER: {m['wer']:.4f}  CER: {m['cer']:.4f}")
    txt_lines.append("=" * 60)
    out_txt = output_dir / "evaluation_report.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(txt_lines))
    logger.info(f"Summary saved to {out_txt}")

    return report


if __name__ == "__main__":
    main()
