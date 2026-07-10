import json
import yaml
from tokenizers import Tokenizer, pre_tokenizers, decoders, models, trainers, AddedToken
from tokenizers.pre_tokenizers import Sequence, ByteLevel, Digits
from pathlib import Path


def buildTokenizer():
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

    tokenizer.train(trainer, [
        "data/fineweb-edu.txt",
        "data/pg19.txt",
        "data/bookcorpus2.txt",
        "data/numina.txt",
        "data/open-web-math.txt",
        "data/wikipedia.txt",
        "data/arxiv.txt",
    ])

    print(f"  Actual vocab size: {tokenizer.get_vocab_size()}")

    outputDir = Path("tokenizer")
    outputDir.mkdir(exist_ok=True)

    tokenizer.save(str(outputDir / "tokenizer.json"))
    print("  Saved to tokenizer/tokenizer.json")

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

    print("  Saved config to tokenizer/tokenizer_config.json")
    print("Done!")


if __name__ == "__main__":
    buildTokenizer()
