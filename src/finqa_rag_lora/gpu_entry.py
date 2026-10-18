"""Entry point for GPU jobs on Vertex / Agent Platform.

Vertex mounts Cloud Storage buckets at /gcs/<bucket>/. This script copies inputs from the bucket
to local disk, runs the task there, and copies results back to the bucket when done (and after
each mode, so a failure later in the job doesn't lose finished work).

Example (what the job runs):
    python -m finqa_rag_lora.gpu_entry eval --config configs/qa_qwen_base.yaml \\
        --modes retrieved oracle --gcs-root /gcs/finqa-rag-lora-data
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml

from finqa_rag_lora import eval_qa
from finqa_rag_lora.generate import make_generator

INPUT_FILES = ["data/corpus.parquet", "data/eval_test.jsonl", "data/eval_dev.jsonl",
               "results/retrieval_test.jsonl", "results/retrieval_dev.jsonl"]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return True


def stage_inputs(gcs_root: Path, work: Path) -> None:
    for rel in INPUT_FILES:
        found = copy_if_exists(gcs_root / rel, work / rel)
        print(f"  {'copied ' if found else 'missing'} {rel}")
    if not (work / "data/corpus.parquet").exists():
        raise FileNotFoundError(f"{gcs_root}/data/corpus.parquet not found: upload the data files first.")


def local_config(config_path: str, work: Path) -> Path:
    """The same config, with every path pointing into the local work directory."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg.update(corpus_path=str(work / "data/corpus.parquet"), eval_dir=str(work / "data"),
               retrieval_dir=str(work / "results"), results_dir=str(work / "results/qa"))
    if cfg.get("adapter") and str(cfg["adapter"]).startswith("gs://"):
        raise ValueError("Use the mounted path for adapters, e.g. /gcs/<bucket>/adapters/<name>, not gs://")
    out = work / "config.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def check_gpu(cfg: dict) -> None:
    """Stop at once if the model would run on the CPU: on a GPU machine that means a setup problem,
    and the job would otherwise crawl along (and bill) for hours. Set require_gpu: false to allow CPU."""
    if cfg.get("generator") != "hf":
        return
    import torch
    available = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if available else "none"
    print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | GPU available: {available} ({name})",
          flush=True)
    if not available and cfg.get("require_gpu", True):
        raise RuntimeError("No GPU visible to PyTorch. Refusing to run on the CPU. Check that the image "
                           "is built on an NVIDIA CUDA base and that the job requested a GPU.")


def run_eval(config_path: str, modes: list[str], split: str, limit: int | None, gcs_root: Path) -> None:
    work = Path(tempfile.mkdtemp(prefix="finqa_"))
    print(f"Staging inputs from {gcs_root} -> {work}")
    stage_inputs(gcs_root, work)
    cfg_path = local_config(config_path, work)
    cfg = yaml.safe_load(cfg_path.read_text())
    missing = [k for k in ("run_name", "generator", "model", "k") if k not in cfg]
    if missing:
        raise ValueError(f"{config_path} is missing required settings {missing}. Is it the right file?")
    check_gpu(cfg)

    # Load the model once and reuse it for every mode.
    cache: dict = {}

    def factory(c: dict):
        if "gen" not in cache:
            cache["gen"] = make_generator(c)
        return cache["gen"]

    for mode in modes:
        run_name = f"{cfg['run_name']}_{mode}_{split}" + (f"_limit{limit}" if limit else "")
        local_run, remote_run = work / "results/qa" / run_name, gcs_root / "results/qa" / run_name
        if copy_if_exists(remote_run, local_run):
            print(f"Resuming {run_name} from saved results in the bucket")
        def save(local_run=local_run, remote_run=remote_run):
            copy_if_exists(local_run, remote_run)

        try:
            eval_qa.main(str(cfg_path), mode, split, limit, overwrite=False, generator_factory=factory,
                         checkpoint=save)
        finally:
            copy_if_exists(local_run, remote_run)  # save whatever finished, even on failure
            print(f"Saved {run_name} -> {remote_run}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="task", required=True)
    ev = sub.add_parser("eval", help="answer the eval questions with a local model")
    ev.add_argument("--config", required=True)
    ev.add_argument("--modes", nargs="+", default=["retrieved", "oracle"], choices=["retrieved", "oracle"])
    ev.add_argument("--split", default="test", choices=["test", "dev"])
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--gcs-root", required=True, help="mounted bucket path, e.g. /gcs/finqa-rag-lora-data")
    args = parser.parse_args()
    if args.task == "eval":
        run_eval(args.config, args.modes, args.split, args.limit, Path(args.gcs_root))


if __name__ == "__main__":
    main()