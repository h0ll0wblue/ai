import json
import os
import argparse
import yaml
from tokenizers import Tokenizer, pre_tokenizers, decoders, models, trainers, AddedToken
from tokenizers.pre_tokenizers import Sequence, ByteLevel, Digits
from pathlib import Path


def buildTokenizer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-datasets", nargs="*", default=[],
        help="Train from HuggingFace datasets instead of local .txt files")
    parser.add_argument("--sample", type=int, default=5000,
        help="Number of examples to sample from each HF dataset for training (5000 = ~10-20MB of text)")
    parser.add_argument("--output", type=str, default="tokenizer",
        help="Output directory")
    args = parser.parse_args()

    with open("configs/tokenizer.yaml") as f:
        cfg = yaml.safe_load(f)

    tokenizer = Tokenizer(models.BPE(unk_token=None))

    tokenizer.pre_tokenizer = Sequence([
        Digits(individual_digits=cfg["pretokenizer"]["individualDigits"]),
        ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True),
    ])

    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = None

    padToken = AddedToken(cfg["specialTokens"]["pad"], special=True)
    bosToken = AddedToken(cfg["specialTokens"]["bos"], special=True)
    eosToken = AddedToken(cfg["specialTokens"]["eos"], special=True)

    trainer = trainers.BpeTrainer(
        vocab_size=cfg["vocabSize"],
        min_frequency=cfg["minFrequency"],
        special_tokens=[padToken, bosToken, eosToken],
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    print("Training tokenizer...")
    print(f"  Vocab size: {cfg['vocabSize']}")
    print(f"  Min frequency: {cfg['minFrequency']}")
    print("  Pre-tokenizer: Digits(individual) + ByteLevel(GPT-2 regex)")

    if args.from_datasets:
        from datasets import load_dataset
        dataFiles = []
        for dsName in args.from_datasets:
            print(f"  Loading dataset: {dsName}...")
            parts = dsName.split(":", 1)
            name = parts[0]
            split = parts[1] if len(parts) > 1 else "train"
            ds = load_dataset(name, split=split, streaming=True)
            textKey = "text" if "text" in ds.features else list(ds.features.keys())[0]
            texts = []
            logInterval = max(1, args.sample // 10)
            for i, example in enumerate(ds):
                if i >= args.sample:
                    break
                texts.append(example[textKey])
                if (i + 1) % logInterval == 0:
                    print(f"    ... loaded {i+1}/{args.sample} examples")
            print(f"    Loaded {len(texts)} examples, saving to temp file...")
            tmpPath = f"/tmp/tokenizer-data-{name.replace('/', '-')}.txt"
            with open(tmpPath, "w", encoding="utf-8") as f:
                for t in texts:
                    f.write(t + "\n")
            dataFiles.append(tmpPath)
        print(f"  Starting BPE training on {len(dataFiles)} file(s)...")
        print(f"  This may take several minutes with no visible progress.")
        tokenizer.train(trainer, dataFiles)
        print(f"  BPE training complete!")
        for p in dataFiles:
            os.remove(p)
    else:
        dataFiles = [
            "data/fineweb-edu.txt",
            "data/pg19.txt",
            "data/bookcorpus2.txt",
            "data/numina.txt",
            "data/open-web-math.txt",
            "data/wikipedia.txt",
            "data/arxiv.txt",
        ]
        missing = [f for f in dataFiles if not os.path.exists(f)]
        if missing:
            print("ERROR: Local data files not found. Either:")
            print("  - Run with --from-datasets to train from HuggingFace datasets")
            print(f"  - Or create these files: {missing}")
            exit(1)
        tokenizer.train(trainer, dataFiles)

    print(f"  Actual vocab size: {tokenizer.get_vocab_size()}")

    outputDir = Path(args.output)
    outputDir.mkdir(exist_ok=True)

    tokenizer.save(str(outputDir / "tokenizer.json"))
    print(f"  Saved to {outputDir / 'tokenizer.json'}")

    hfConfig = {
        "add_prefix_space": False,
        "bos_token": cfg["specialTokens"]["bos"],
        "eos_token": cfg["specialTokens"]["eos"],
        "pad_token": cfg["specialTokens"]["pad"],
        "model_max_length": 4096,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
    }
    with open(outputDir / "tokenizer_config.json", "w") as f:
        json.dump(hfConfig, f, indent=2)

    print(f"  Saved config to {outputDir / 'tokenizer_config.json'}")
    print("Done!")


if __name__ == "__main__":
    buildTokenizer()
