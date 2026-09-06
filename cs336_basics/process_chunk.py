import regex as re
import multiprocessing
import os
from typing import BinaryIO

def process_chunk(
    file_path : str,
    start : int,
    end : int,
    special_tokens : list[bytes]
):
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    text = chunk_bytes.decode("utf-8", errors="ignore")
  
    # A very cool tokenizer (GPT-2)
    reg = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    ### THIS SPLITS ON SPECIAL TOKENS ###
    
    special_token_strs = []
    for i, special_token in enumerate(special_tokens):
        special_token_str = special_token.decode("utf-8")
        special_token_strs.append(re.escape(special_token_str))
    
    split_pattern = '|'.join(special_token_strs)
    docs = re.split(split_pattern, text)

    pretokens = {}

    # Counts how man ytimes each pretoken string occurs
    for doc in docs:
        for match in re.finditer(reg, doc):
            word = match.group()
            word_bytes = word.encode("utf-8") # encode pretoken into bytes
            pretoken_tuple = ()

            for byte in word_bytes:
                pretoken_tuple += (bytes([byte]),)

            pretokens[pretoken_tuple] = pretokens.get(pretoken_tuple, 0) + 1

    return pretokens