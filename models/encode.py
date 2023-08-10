import sys
from pathlib import Path

from tokenizers import Tokenizer

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <PROMPT>")
    exit(1)
tokenizer = Tokenizer.from_file(str(Path(__file__).parent / "tokenizer.json"))
print(tokenizer.encode(sys.argv[1]).ids)
