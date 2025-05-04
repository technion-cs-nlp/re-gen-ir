from transformer_lens import HookedEncoderDecoder
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from tqdm import tqdm
import json

from collections import defaultdict

from jaxtyping import Float
import torch
torch.set_grad_enabled(False)
from transformers import AutoModelForSeq2SeqLM
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
from os import makedirs

import argparse

parser = argparse.ArgumentParser(description='Swap Encoders')

parser.add_argument('--complete-model-path', type=str, help='Path to the complete model checkpoint')
parser.add_argument('--missing-docs-for-model', type=str, help='Path to the missing documents file. The file should contain a JSON dictionary, where the keys are model paths and the values are comma-separated strings of document IDs.')
parser.add_argument('--val-queries', type=str, help='Path to the valid queries file')
parser.add_argument('--correct-valid', type=str, help='path to the correct_valid.json file', default='[checkpoint]/correct_valid.json')
parser.add_argument('-o', '--output-path', type=str, help='path to which the output files are written', default='[checkpoint]')


args = parser.parse_args()

complete_model_path = args.complete_model_path
missing_docs_for_model = json.load(open(args.missing_docs_for_model))
valid_queries = json.load(open(args.val_queries))


def get_tokens(text, tokenizer, pad=None):
    if not pad:
        return tokenizer(text, return_tensors='pt')['input_ids']
    else:
        return tokenizer(text, return_tensors='pt', padding='max_length', max_length=pad)['input_ids']

def tokens_to_str(tokens, tokenizer):
    return [tokenizer.decode(d) for d in tokens[0]]

decoder_input = torch.tensor([[0]])

def get_ranks(tensor):
    return torch.argsort(torch.argsort(tensor, descending=True))

correct_valid_path = args.correct_valid.replace('[checkpoint]', complete_model_path)
correct_valid_complete = json.load(open(correct_valid_path))

q_id_to_q_complete = {q['id']:q['query'] for q in valid_queries}
correct_queries_with_d_id_complete = [(q_id_to_q_complete[q], d) for d,q in correct_valid_complete]

device = utils.get_device()

def load_model(model_path):
    
    OFFICIAL_MODEL_NAMES.append(model_path)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return HookedEncoderDecoder.from_pretrained(model_path, hf_model=hf_model, device=device)

tokenizer = AutoTokenizer.from_pretrained(complete_model_path)
complete_model = load_model(complete_model_path)
results = defaultdict(dict)
for missing_model_path in missing_docs_for_model:
    print(missing_model_path)
    missing_model = load_model(missing_model_path)
    def get_new_logits(input):
        input_tokens = get_tokens(input, tokenizer)
        missing_logits, missing_cache = missing_model.run_with_cache(input_tokens, decoder_input, remove_batch_dim=True)

        def missing_patching_hook(
            emb: Float[torch.Tensor, "batch pos d_model"],
            hook: HookPoint,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
            # Each HookPoint has a name attribute giving the name of the hook.
            missing_emb = missing_cache[hook.name]
            emb[:, :] = missing_emb[:, :]
            return emb


        input_tokens = get_tokens(input, tokenizer) 
        swapped_logits = complete_model.run_with_hooks(input_tokens, decoder_input=decoder_input, fwd_hooks=[
            ("encoder.23.hook_resid_post", missing_patching_hook)
        ])
        return swapped_logits

    rank_of_prev_correct_missing_model = []
    for query, correct_doc in tqdm(correct_queries_with_d_id_complete):
        swapped_logits = get_new_logits(query)
        correct_doc_id = tokenizer.convert_tokens_to_ids(correct_doc)
        new_rank = get_ranks(swapped_logits)[0][0][correct_doc_id]
        rank_of_prev_correct_missing_model.append(new_rank.item())

    results[missing_model_path]['Mean Rank of prev. correct'] = sum(rank_of_prev_correct_missing_model) / len(rank_of_prev_correct_missing_model)
    results[missing_model_path]['Median:'] = torch.median(torch.tensor(rank_of_prev_correct_missing_model, dtype=torch.float)).item()
    results[missing_model_path]['Number of non 0 ranks'] = len([r for r in rank_of_prev_correct_missing_model if r != 0])
    results[missing_model_path]['Non 0 ranks'] = [r for r in rank_of_prev_correct_missing_model if r != 0]
    
    queries, docs = zip(*correct_queries_with_d_id_complete)

    missing_ranks = []
    if type(missing_docs_for_model[missing_model_path]) == str:
        missing_docs = missing_docs_for_model[missing_model_path].split(',')
    else:
        missing_docs = missing_docs_for_model[missing_model_path]
    for missing_doc in missing_docs:
        missing_doc = missing_doc.strip()
        missing_doc_idx = docs.index(missing_doc)
        print('Missing IDX', missing_doc, missing_doc_idx)
        missing_ranks.append(rank_of_prev_correct_missing_model[missing_doc_idx])
    results[missing_model_path]['missing_ranks'] = missing_ranks
    results[missing_model_path]["Mean of Missing document ranks"] = sum(missing_ranks)/len(missing_ranks)
    rr = 0
    for rank in missing_ranks:
        rr += 1/(rank+1)
    results[missing_model_path]["MRR:"] = rr/len(missing_ranks)
    results[missing_model_path]["median"] = torch.median(torch.tensor(missing_ranks, dtype=torch.float)).item()


out_dir = args.output_path.replace('[checkpoint]', args.checkpoint)
makedirs(out_dir, exist_ok=True)

json.dump(results, open(out_dir + '/swapped_encoder.json'))

