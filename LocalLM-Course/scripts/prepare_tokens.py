import argparse
import json
import sys
from array import array
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, decoders, models
from tokenizers import pre_tokenizers, trainers

from prepare_data import ROOT, digest
from prepare_data import verify as verify_data


DATA = ROOT / "data" / "prepared"
OUT = ROOT / "data" / "tokenized"
VOCAB_SIZE = 4096
EOT = "<|endoftext|>"
VERSION = "0.22.2"


def read_texts(name):
    with (DATA / f"{name}.jsonl").open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            text = json.loads(line)["text"]
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"Invalid text in {name}")
            if EOT in text:
                raise RuntimeError("Reserved marker in story text.")
            yield text


def check_tokenizer(tokenizer):
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise RuntimeError("Expected a vocabulary of 4096.")
    if tokenizer.token_to_id(EOT) != 0:
        raise RuntimeError("Expected end-of-text ID 0.")
    probes = [
        "The little dog found a red ball.",
        "  Spaces stay.  ",
        "A caf\u00e9 near the river.",
        "quizzledragon 12345\nSecond line.",
    ]
    for text in probes:
        ids = tokenizer.encode(
            text, add_special_tokens=False
        ).ids
        decoded = tokenizer.decode(
            ids, skip_special_tokens=False
        )
        if decoded != text:
            raise RuntimeError("Tokenizer round trip failed.")


def encode_split(tokenizer, name, expected_stories):
    path = OUT / f"{name}.bin"
    eot_id = tokenizer.token_to_id(EOT)
    stories = 0
    tokens = 0

    with path.open("wb") as stream:
        for text in read_texts(name):
            ids = tokenizer.encode(
                text, add_special_tokens=False
            ).ids
            if not ids or eot_id in ids:
                raise RuntimeError("Unexpected story encoding.")
            if min(ids) < 0 or max(ids) >= VOCAB_SIZE:
                raise RuntimeError("Token ID is out of range.")
            if tokenizer.decode(ids) != text:
                raise RuntimeError("Story round trip failed.")
            ids.append(eot_id)
            values = array("H", ids)
            if sys.byteorder != "little":
                values.byteswap()
            values.tofile(stream)
            stories += 1
            tokens += len(ids)

    if stories != expected_stories:
        raise RuntimeError(f"Story count differs: {name}")

    print(f"Encoded {name}: {tokens:,} tokens", flush=True)
    return {
        "stories": stories,
        "tokens": tokens,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def verify():
    verify_data()
    manifest = json.loads(
        (OUT / "manifest.json").read_text(encoding="utf-8")
    )
    if digest(DATA / "manifest.json") != manifest[
        "data_manifest_sha256"
    ]:
        raise RuntimeError("The source data manifest changed.")

    tokenizer_path = OUT / "tokenizer.json"
    if digest(tokenizer_path) != manifest["tokenizer_sha256"]:
        raise RuntimeError("The saved tokenizer changed.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    check_tokenizer(tokenizer)

    for name in ("train", "val", "test"):
        path = OUT / f"{name}.bin"
        expected = manifest["splits"][name]
        if digest(path) != expected["sha256"]:
            raise RuntimeError(f"Token file changed: {name}")
        if path.stat().st_size != 2 * expected["tokens"]:
            raise RuntimeError(f"Wrong byte count: {name}")

        values = array("H")
        values.frombytes(path.read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        if len(values) < 257 or max(values) >= VOCAB_SIZE:
            raise RuntimeError(f"Invalid token stream: {name}")
        if values[-1] != 0:
            raise RuntimeError(f"Missing final boundary: {name}")
        if values.count(0) != expected["stories"]:
            raise RuntimeError(f"Wrong boundary count: {name}")
        print(f"{name}: {len(values):,} tokens verified")

    print("Vocabulary: 4096")
    print("End-of-text ID: 0")
    print("Token verification passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if tokenizers.__version__ != VERSION:
        raise RuntimeError("Install tokenizers==0.22.2.")
    if array("H").itemsize != 2:
        raise RuntimeError("This script requires 2-byte shorts.")
    if args.verify:
        verify()
        return
    if OUT.exists():
        raise RuntimeError(
            "Tokenized folder already exists. Use --verify, "
            "or preserve it before rebuilding."
        )

    verify_data()
    data_manifest = json.loads(
        (DATA / "manifest.json").read_text(encoding="utf-8")
    )
    OUT.mkdir(parents=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[EOT],
        show_progress=True,
    )
    print("Training tokenizer on train.jsonl only.", flush=True)
    tokenizer.train_from_iterator(
        read_texts("train"),
        trainer=trainer,
        length=data_manifest["splits"]["train"]["stories"],
    )
    check_tokenizer(tokenizer)

    tokenizer_path = OUT / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    check_tokenizer(tokenizer)

    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = encode_split(
            tokenizer,
            name,
            data_manifest["splits"][name]["stories"],
        )

    manifest = {
        "format": "uint16-le",
        "vocab_size": VOCAB_SIZE,
        "end_of_text_id": 0,
        "boundary_rule": "One end token after every story.",
        "tokenizer_training_split": "train",
        "tokenizers_version": tokenizers.__version__,
        "python_version": sys.version,
        "data_manifest_sha256": digest(DATA / "manifest.json"),
        "tokenizer_sha256": digest(tokenizer_path),
        "preparation_script_sha256": digest(Path(__file__)),
        "splits": splits,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    verify()


if __name__ == "__main__":
    main()
