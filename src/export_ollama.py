"""Export the trained SLM to GGUF and generate an Ollama Modelfile.

    python src/export_ollama.py                    # writes models/ollama/*.gguf + Modelfile
    python src/export_ollama.py --dtype f32
    python src/export_ollama.py --no-graph

Our model is a bias-free, RMSNorm + RoPE + SwiGLU decoder, which is exactly the
``llama`` architecture in llama.cpp. The exporter therefore rewrites the
checkpoint into llama.cpp tensor names and writes a GGUF v3 file by hand (no
third-party GGUF library required), together with the tokenizer in GPT-2 BPE
form so Ollama can tokenize text identically to training.

Requires Ollama to be installed for ``ollama create``; this script only produces
the artifacts. Validate the result with ``--verify`` (needs `pip install gguf`).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import ModelConfig, build_model, load_checkpoint  # noqa: E402
from tokenizer import ByteLevelBPETokenizer  # noqa: E402

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12
# GGUF tensor types
GGML_TYPE_F32, GGML_TYPE_F16 = 0, 1
# llama.cpp token types
TOKEN_TYPE_NORMAL, TOKEN_TYPE_CONTROL = 1, 3


# --------------------------------------------------------------------------- #
# GPT-2 byte <-> unicode mapping (required by llama.cpp's gpt2 tokenizer)
# --------------------------------------------------------------------------- #
def bytes_to_unicode() -> Dict[int, str]:
    """The reversible byte->unicode table used by GPT-2 style BPE tokenizers."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_UNICODE = bytes_to_unicode()


# --------------------------------------------------------------------------- #
# GGUF writer
# --------------------------------------------------------------------------- #
class GGUFWriter:
    """Minimal GGUF v3 writer: metadata key/values followed by tensor data."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.kv: List[Tuple[str, int, Any]] = []
        self.tensors: List[Tuple[str, np.ndarray]] = []

    # ---- metadata ----
    def add(self, key: str, value_type: int, value: Any) -> None:
        self.kv.append((key, value_type, value))

    def add_string(self, key: str, value: str) -> None:
        self.add(key, T_STRING, value)

    def add_u32(self, key: str, value: int) -> None:
        self.add(key, T_UINT32, int(value))

    def add_f32(self, key: str, value: float) -> None:
        self.add(key, T_FLOAT32, float(value))

    def add_bool(self, key: str, value: bool) -> None:
        self.add(key, T_BOOL, bool(value))

    def add_string_array(self, key: str, values: List[str]) -> None:
        self.add(key, T_ARRAY, (T_STRING, values))

    def add_i32_array(self, key: str, values: List[int]) -> None:
        self.add(key, T_ARRAY, (T_INT32, list(values)))

    # ---- tensors ----
    def add_tensor(self, name: str, array: np.ndarray) -> None:
        self.tensors.append((name, array))

    # ---- serialisation ----
    @staticmethod
    def _pack_string(value: str) -> bytes:
        data = value.encode("utf-8")
        return struct.pack("<Q", len(data)) + data

    def _pack_value(self, value_type: int, value: Any) -> bytes:
        if value_type == T_STRING:
            return self._pack_string(value)
        if value_type == T_UINT32:
            return struct.pack("<I", int(value))
        if value_type == T_INT32:
            return struct.pack("<i", int(value))
        if value_type == T_FLOAT32:
            return struct.pack("<f", float(value))
        if value_type == T_BOOL:
            return struct.pack("<?", bool(value))
        if value_type == T_ARRAY:
            element_type, items = value
            out = struct.pack("<IQ", element_type, len(items))
            if element_type == T_STRING:
                return out + b"".join(self._pack_string(str(i)) for i in items)
            if element_type == T_INT32:
                return out + b"".join(struct.pack("<i", int(i)) for i in items)
            raise ValueError(f"Unsupported array element type {element_type}")
        raise ValueError(f"Unsupported value type {value_type}")

    def write(self, alignment: int = 32) -> Dict[str, Any]:
        dtype = np.float16 if all(t[1].dtype == np.float16 for t in self.tensors) else np.float32
        ggml_type = GGML_TYPE_F16 if dtype == np.float16 else GGML_TYPE_F32

        header = bytearray()
        header += GGUF_MAGIC
        header += struct.pack("<I", GGUF_VERSION)
        header += struct.pack("<Q", len(self.tensors))
        header += struct.pack("<Q", len(self.kv))
        for key, value_type, value in self.kv:
            header += self._pack_string(key)
            header += struct.pack("<I", value_type)
            header += self._pack_value(value_type, value)

        # tensor infos + data offsets, relative to the start of the data section
        infos = bytearray()
        offset = 0
        payloads: List[bytes] = []
        for name, array in self.tensors:
            array = np.ascontiguousarray(array.astype(dtype))
            payload = array.tobytes()
            infos += self._pack_string(name)
            infos += struct.pack("<I", array.ndim)
            for dim in reversed(array.shape):  # GGUF stores fastest-varying dim first
                infos += struct.pack("<Q", int(dim))
            infos += struct.pack("<I", ggml_type)
            infos += struct.pack("<Q", offset)
            pad = (-len(payload)) % alignment
            payloads.append(payload + b"\x00" * pad)
            offset += len(payload) + pad

        header += infos
        pad = (-len(header)) % alignment
        header += b"\x00" * pad

        with open(self.path, "wb") as fh:
            fh.write(header)
            for payload in payloads:
                fh.write(payload)
        return {
            "path": self.path,
            "size_bytes": os.path.getsize(self.path),
            "tensor_count": len(self.tensors),
            "metadata_keys": len(self.kv),
            "dtype": "float16" if dtype == np.float16 else "float32",
            "ggml_type": ggml_type,
            "data_alignment": alignment,
        }


# --------------------------------------------------------------------------- #
# Checkpoint -> llama-arch tensors
# --------------------------------------------------------------------------- #
def checkpoint_to_llama_tensors(state: Dict[str, torch.Tensor], config: ModelConfig,
                                dtype: np.dtype) -> List[Tuple[str, np.ndarray]]:
    """Rename and split our state_dict into llama.cpp tensor layout."""
    tensors: List[Tuple[str, np.ndarray]] = []

    def add(name: str, tensor: torch.Tensor) -> None:
        tensors.append((name, tensor.detach().to(torch.float32).cpu().numpy().astype(dtype)))

    add("token_embd.weight", state["token_embedding.weight"])
    add("output_norm.weight", state["norm_f.weight"])
    if "lm_head.weight" in state and not config.tie_weights:
        add("output.weight", state["lm_head.weight"])
    else:
        # tied embeddings: llama.cpp can use token_embd as the output matrix,
        # but writing it explicitly avoids ambiguity across versions.
        add("output.weight", state["token_embedding.weight"])

    for layer in range(config.n_layer):
        prefix = f"blocks.{layer}."
        qkv = state[prefix + "attn.qkv.weight"]
        q, k, v = qkv.chunk(3, dim=0)
        add(f"blk.{layer}.attn_norm.weight", state[prefix + "norm1.weight"])
        add(f"blk.{layer}.attn_q.weight", q)
        add(f"blk.{layer}.attn_k.weight", k)
        add(f"blk.{layer}.attn_v.weight", v)
        add(f"blk.{layer}.attn_output.weight", state[prefix + "attn.proj.weight"])
        add(f"blk.{layer}.ffn_norm.weight", state[prefix + "norm2.weight"])
        add(f"blk.{layer}.ffn_gate.weight", state[prefix + "mlp.w1.weight"])
        add(f"blk.{layer}.ffn_up.weight", state[prefix + "mlp.w3.weight"])
        add(f"blk.{layer}.ffn_down.weight", state[prefix + "mlp.w2.weight"])
    return tensors


def build_gguf(state: Dict[str, torch.Tensor], config: ModelConfig, tokenizer: ByteLevelBPETokenizer,
               out_path: str, dtype: np.dtype, model_name: str) -> Dict[str, Any]:
    writer = GGUFWriter(out_path)
    ff_hidden = int(state["blocks.0.mlp.w1.weight"].shape[0])

    writer.add_string("general.architecture", "llama")
    writer.add_string("general.name", model_name)
    writer.add_u32("general.file_type", GGML_TYPE_F16 if dtype == np.float16 else GGML_TYPE_F32)
    writer.add_string("general.description",
                      "Air Quality SLM: small decoder-only Transformer trained on air quality records")
    writer.add_u32("llama.context_length", config.block_size)
    writer.add_u32("llama.embedding_length", config.n_embd)
    writer.add_u32("llama.block_count", config.n_layer)
    writer.add_u32("llama.feed_forward_length", ff_hidden)
    writer.add_u32("llama.attention.head_count", config.n_head)
    writer.add_u32("llama.attention.head_count_kv", config.n_head)
    writer.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_u32("llama.rope.dimension_count", config.n_embd // config.n_head)
    writer.add_f32("llama.rope.freq_base", 10000.0)

    # ---- tokenizer in GPT-2 BPE form ----
    tokens: List[str] = []
    token_types: List[int] = []
    n_special = len(tokenizer.special_tokens)
    for token_id in range(tokenizer.vocab_size):
        if token_id < n_special:
            tokens.append(tokenizer.special_tokens[token_id])
            token_types.append(TOKEN_TYPE_CONTROL)
            continue
        raw = tokenizer.id_to_bytes.get(token_id)
        if raw is None:
            tokens.append("")
            token_types.append(TOKEN_TYPE_CONTROL)
            continue
        tokens.append("".join(BYTE_TO_UNICODE[b] for b in raw))
        token_types.append(TOKEN_TYPE_NORMAL)

    merges: List[str] = []
    for left, right in tokenizer.merges:
        left_str = "".join(BYTE_TO_UNICODE[b] for b in tokenizer.id_to_bytes[left])
        right_str = "".join(BYTE_TO_UNICODE[b] for b in tokenizer.id_to_bytes[right])
        merges.append(f"{left_str} {right_str}")

    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "gpt2")
    writer.add_string_array("tokenizer.ggml.tokens", tokens)
    writer.add_i32_array("tokenizer.ggml.token_type", token_types)
    writer.add_string_array("tokenizer.ggml.merges", merges)
    writer.add_u32("tokenizer.ggml.bos_token_id", tokenizer.bos_id)
    writer.add_u32("tokenizer.ggml.eos_token_id", tokenizer.eos_id)
    writer.add_u32("tokenizer.ggml.padding_token_id", tokenizer.pad_id)
    writer.add_bool("tokenizer.ggml.add_bos_token", True)
    writer.add_bool("tokenizer.ggml.add_eos_token", False)

    for name, array in checkpoint_to_llama_tensors(state, config, dtype):
        writer.add_tensor(name, array)

    info = writer.write()
    info["tensors"] = [name for name, _ in writer.tensors]
    return info


MODELFILE_TEMPLATE = """FROM {gguf_name}

# Prompt format used during training. Keep it identical or answers degrade.
TEMPLATE \"\"\"### Question and answer
Question: {{{{ .Prompt }}}}
Answer:\"\"\"

SYSTEM \"\"\"You are an Air Quality assistant. Answer using air quality measurements:
pollutant concentrations, weather context, locations and air quality categories.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 128
PARAMETER stop "### End"
PARAMETER stop "\\n###"
PARAMETER stop "\\nQuestion:"
"""


def write_modelfile(path: str, gguf_name: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(MODELFILE_TEMPLATE.format(gguf_name=gguf_name))


def verify_gguf(path: str) -> Dict[str, Any]:
    """Read the file back with the reference `gguf` package, if installed."""
    try:
        import gguf  # type: ignore
    except ImportError:
        return {"verified": False, "reason": "python package 'gguf' not installed (pip install gguf)"}
    reader = gguf.GGUFReader(path)
    fields = {}
    for name, field in reader.fields.items():
        if name.startswith("tokenizer."):
            values = getattr(field, "data", None)
            fields[name] = f"<{len(values)} items>" if values is not None and hasattr(values, "__len__") else "<field>"
        else:
            try:
                fields[name] = field.contents()
            except Exception:
                fields[name] = repr(field)
    tensors = {t.name: [int(d) for d in t.shape] for t in reader.tensors}
    version = getattr(reader, "version", None) or getattr(gguf, "GGUF_VERSION", None)
    return {"verified": True, "version": version, "metadata": fields,
            "tensor_count": len(tensors), "tensors": tensors}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export the Air Quality SLM to GGUF + Ollama Modelfile.")
    parser.add_argument("--models-dir", default=os.path.join(PROJECT_ROOT, "models", "air_quality_slm"))
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "models", "ollama"))
    parser.add_argument("--weights", default="best_model.pt", help="checkpoint file inside --models-dir")
    parser.add_argument("--dtype", default="f16", choices=["f16", "f32"],
                        help="f16 halves the file size; f32 is safer for debugging")
    parser.add_argument("--name", default="air-quality-slm")
    parser.add_argument("--verify", action="store_true", help="read the GGUF back to validate the format")
    parser.add_argument("--no-graph", action="store_true", help="skip the Modelfile")
    args = parser.parse_args(argv)

    weights = os.path.join(args.models_dir, args.weights)
    if not os.path.exists(weights):
        print(f"ERROR: no checkpoint at {weights}\nTrain first: python src/train_slm.py")
        return 2

    tokenizer = ByteLevelBPETokenizer.load(os.path.join(args.models_dir, "tokenizer"))
    payload = load_checkpoint(weights, map_location="cpu")
    config = ModelConfig.from_dict(payload["config"])
    if config.vocab_size != tokenizer.vocab_size:
        print(f"ERROR: checkpoint vocab {config.vocab_size} != tokenizer vocab {tokenizer.vocab_size}")
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    gguf_name = f"{args.name}.{args.dtype}.gguf"
    out_path = os.path.join(args.out_dir, gguf_name)
    dtype = np.float16 if args.dtype == "f16" else np.float32

    print(f"Converting {weights}")
    print(f"  arch=llama  layers={config.n_layer} heads={config.n_head} "
          f"d_model={config.n_embd} ctx={config.block_size} vocab={config.vocab_size}")
    info = build_gguf(payload["model_state"], config, tokenizer, out_path, dtype, args.name)
    print(f"  wrote {info['path']} ({info['size_bytes'] / 1024 ** 2:.2f} MB, "
          f"{info['tensor_count']} tensors, {info['metadata_keys']} metadata keys, {info['dtype']})")

    modelfile = os.path.join(args.out_dir, "Modelfile")
    if not args.no_graph:
        write_modelfile(modelfile, gguf_name)
        print(f"  wrote {modelfile}")

    summary = {"gguf": info, "modelfile": None if args.no_graph else modelfile,
               "ollama_name": args.name, "config": config.to_dict()}
    if args.verify:
        result = verify_gguf(out_path)
        summary["verification"] = result
        if result.get("verified"):
            print(f"  VERIFIED: GGUF v{result['version']}, {result['tensor_count']} tensors readable")
            for key in ("general.architecture", "llama.block_count", "llama.embedding_length"):
                if key in result["metadata"]:
                    print(f"    {key} = {result['metadata'][key]}")
        else:
            print(f"  NOT VERIFIED: {result.get('reason')}")

    with open(os.path.join(args.out_dir, "export_report.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\nNext (requires Ollama installed):")
    print(f"  cd {args.out_dir}")
    print(f"  ollama create {args.name} -f Modelfile")
    print(f"  ollama run {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
