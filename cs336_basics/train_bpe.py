import regex as re
import multiprocessing
import os
from typing import BinaryIO
from find_chunk_boundaries import find_chunk_boundaries
from process_chunk import process_chunk

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
):

    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for spec_tok in special_tokens:
        vocab[len(vocab)] = spec_tok.encode("utf-8")

    merges = []

    # Multi processing here
    num_processes = 8

    bytes_special_tokens = [spec.encode("utf-8") for spec in special_tokens]

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, bytes_special_tokens)

    chunk_args =  [
        (input_path, start, end, bytes_special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_chunk, chunk_args)

    # Merging all the results from different threads
    pretokens = {}
    for result in results:
        for token, count in result.items():
            pretokens[token] = pretokens.get(token, 0) + count
    

    ### MERGING OF FREQUENT TOKEN PAIRS STARTS HERE ###
    target = vocab_size - len(vocab)
    

    for i in range(target):

        max = 0
        top = ()
        frequency = {}
        # Calculates frequency of token pairs
        for pretoken in pretokens:
            for k in range(len(pretoken)-1):
                pair = (pretoken[k], pretoken[k+1])

                if pair not in frequency:
                    frequency[pair] = 1
                else:
                    frequency[pair] += 1

        # Finds the most frequent token pair to add to vocab
        for pair in frequency:
            if frequency[pair] > max:
                top = pair
                max = frequency[pair]
            elif frequency[pair] == max and pair > top:
                top = pair

        # Adds most frequence pair to the special tokens and vocab
        merges.append(top)
        special = top[0] + top[1]
        vocab[i + 256] = special

        # Updating the pretokens
        merged_pretokens = {}
        
        for pretoken, count in pretokens.items():
            pretoken_list = list(pretoken)
            n = 0
            while n < len(pretoken)-1:
                pair = (pretoken_list[n], pretoken_list[n+1])
                if pair == top:
                    pretoken_list[n:n+2] = [special]
                else:
                    n += 1
            new_token = tuple(pretoken_list)
            merged_pretokens[new_token] = count

        pretokens = merged_pretokens

    ### MERGING ENDS HERE

    return vocab, merges
        



    
    