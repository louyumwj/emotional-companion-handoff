"""Check dependencies and local model files; failures return a nonzero exit code."""
import argparse
import importlib
import json
from pathlib import Path
import sys

WS = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true", help="Only verify dependencies")
    ap.add_argument("--models", nargs="+", default=["chinese-roberta-wwm-ext", "Qwen2.5-0.5B-Instruct"])
    args = ap.parse_args()
    failed = False
    for name in ["torch", "transformers", "peft", "accelerate", "numpy", "fastapi", "uvicorn", "httpx"]:
        try:
            module = importlib.import_module(name)
            print(f"{name}: OK {getattr(module, '__version__', '')}")
        except Exception as exc:
            print(f"{name}: FAILED {type(exc).__name__}: {exc}")
            failed = True
    if failed or args.skip_models:
        return int(failed)
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from safetensors import safe_open
    for name in args.models:
        path = WS / "models" / name
        try:
            if not path.is_dir():
                raise FileNotFoundError(f"Model directory is missing: {path}")
            config = AutoConfig.from_pretrained(str(path), local_files_only=True)
            tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            tok("verification", add_special_tokens=False)
            index = path / "model.safetensors.index.json"
            if index.is_file():
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                weights = [path / shard for shard in shards]
            elif (path / "model.safetensors").is_file():
                weights = [path / "model.safetensors"]
            else:
                weights = [path / "pytorch_model.bin"]
            for weight in weights:
                if not weight.is_file() or not weight.stat().st_size:
                    raise FileNotFoundError(f"Missing or empty weight: {weight}")
                if weight.suffix == ".safetensors":
                    with safe_open(str(weight), framework="pt", device="cpu") as tensors:
                        if not list(tensors.keys()):
                            raise ValueError(f"No tensors in {weight}")
            if config.model_type == "bert":
                AutoModel.from_pretrained(str(path), local_files_only=True)
            print(f"{name}: OK, tokenizer and {len(weights)} weight file(s)")
        except Exception as exc:
            print(f"{name}: FAILED {type(exc).__name__}: {exc}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
