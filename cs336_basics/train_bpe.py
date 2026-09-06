import regex as re
import multiprocessing
import os
from typing import BinaryIO
from .find_chunk_boundaries import find_chunk_boundaries
from .process_chunk import process_chunk


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
        results = pool.starmap(process_chunk, chunk_args)

    # Merging all the results from different threads
    pretokens = {}
    for result in results:
        for token, count in result.items():
            pretokens[token] = pretokens.get(token, 0) + count
    

    ### MERGING OF FREQUENT TOKEN PAIRS STARTS HERE (ALSO CANT PARALLELIZE THIS) ###
    target = vocab_size - len(vocab)

    frequency = {}
    reverse_index = {}
    # Calculates frequency of token pairs and maps a pair to the set of pretokens 
    # is in.
    for pretoken, count in pretokens.items():
        for k in range(len(pretoken)-1):
            pair = (pretoken[k], pretoken[k+1])

            if pair not in frequency:
                frequency[pair] = 0
                reverse_index[pair] = set()
            
            frequency[pair] += count

            reverse_index[pair].add(pretoken)

    # Maps a token pair to 
    
    for i in range(target):

        max = 0
        top = ()

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
        vocab[len(vocab)] = special

        # Optimized frequency updater here.
        affected_pretokens = list(reverse_index.get(top, ()))

        for pretoken in affected_pretokens:
            count = pretokens[pretoken]
            pretoken_list = list(pretoken)

            # Remove this pretoken's OLD pair contributions
            for k in range(len(pretoken_list)-1):
                pair = (pretoken_list[k], pretoken_list[k+1])
                frequency[pair] -= count
                if frequency[pair] <= 0:
                    del frequency[pair]
                reverse_index[pair].discard(pretoken)

            # Actually perform the merge on this pretoken (handles overlaps)
            n = 0
            while n < len(pretoken_list)-1:
                if (pretoken_list[n], pretoken_list[n+1]) == top:
                    pretoken_list[n:n+2] = [special]
                else:
                    n += 1
            new_pretoken = tuple(pretoken_list)

            # Swap the key in pretokens
            del pretokens[pretoken]
            pretokens[new_pretoken] = pretokens.get(new_pretoken, 0) + count

            # Add this pretoken's NEW pair contributions
            for k in range(len(new_pretoken)-1):
                pair = (new_pretoken[k], new_pretoken[k+1])
                if pair not in frequency:
                    frequency[pair] = 0
                    reverse_index[pair] = set()
                frequency[pair] += count
                reverse_index[pair].add(new_pretoken)

        reverse_index.pop(top, None)
        
        
    ### MERGING ENDS HERE

    return vocab, merges
        



    
    