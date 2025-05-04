from transformer_lens import HookedEncoderDecoder
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES


from argparse import ArgumentParser
from jaxtyping import Float
from functools import partial
from tqdm import tqdm
import json
import pandas as pd
from collections import defaultdict
from evaluate import load
from os import makedirs
import numpy as np

trec_eval = load("trec_eval")


import torch 
torch.set_grad_enabled(False)

parser = ArgumentParser(prog='Run patching')
parser.add_argument('-c', '--checkpoint', help='checkpoint for the model to evaluate')
parser.add_argument('--test-queries', help='path for the queries to evaluate')
parser.add_argument('--qrels', help='path for the test query qrels')
parser.add_argument('--val-queries', help='path for the validation queries, for calculating the mean')
parser.add_argument('--correct-valid', type=str, help='path to the correct_valid.json file', default='[checkpoint]/correct_valid.json')
parser.add_argument('-o', '--output-path', type=str, help='path to which the output files are written', default='[checkpoint]')



def store_results(results, file_name):


    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    json.dump(results, open(file_name, 'w'), cls=NpEncoder)



def removal_patching_hook(
    emb: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
) -> Float[torch.Tensor, "batch pos d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    
    emb[:, :] = torch.zeros_like(emb)
    return emb


def get_ranks(tensor):
    return torch.argsort(torch.argsort(tensor, descending=True))

def get_tokens(text, pad=None):
    if not pad:
        return tokenizer(text, return_tensors='pt')['input_ids']
    else:
        return tokenizer(text, return_tensors='pt', padding='max_length', max_length=pad)['input_ids']

def tokens_to_str(tokens):
    return [tokenizer.decode(d) for d in tokens[0]]

decoder_input = torch.tensor([[0]])




if __name__ == "__main__":

    args = parser.parse_args()


    # Load Model into Transformer Lens

    checkpoint = args.checkpoint
    OFFICIAL_MODEL_NAMES.append(checkpoint)

    hf_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    device = utils.get_device()
    model = HookedEncoderDecoder.from_pretrained(checkpoint, hf_model=hf_model, device=device)

    # Load eval data
    correct_valid_path = args.correct_valid.replace('[checkpoint]', checkpoint)
    correct_valid = json.load(open(correct_valid_path))    
    valid_queries = json.load(open(args.val_queries))
    test_queries = json.load(open(args.test_queries)) #val_queries-25-6959.json
    qrels = json.load(open(args.qrels))

    q_id_to_q = {q['id']:q['query'] for q in test_queries}
    q_id_to_q_valid = {q['id']:q['query'] for q in valid_queries}
    correct_queries_with_d_id = [(q, q_id_to_q_valid[q], d) for d,q in correct_valid]
    queries_with_d_id = [(q['id'], q['query'], "") for q in test_queries]
    

    stages = {
        0: [0,1,2,3,4,5,6],
        1: [7,8,9,10,11,12,13,14,15,16,17],
        2: [18,19,20,21,22,23]
    }

    lengths = []
    for q_id, query, doc in correct_queries_with_d_id:
        lengths.append(len(tokenizer.tokenize(query)))

    max_q_len = max(lengths)


    complete_cache = {}
    mean_cache = {}
    for q_id, query, doc in tqdm(correct_queries_with_d_id):
        input_tokens = get_tokens(query, pad=max_q_len+1)
        t5_logits, t5_cache = model.run_with_cache(input_tokens, decoder_input, remove_batch_dim=True)
        #complete_cache.append(t5_cache)
        for key in t5_cache:
            if key not in complete_cache:
                complete_cache[key] = t5_cache[key]
            else:
                complete_cache[key] += t5_cache[key]

    mean_cache = {}
    for hook in complete_cache:
        #mean_cache_for_hook = torch.zeros_like(complete_cache[0][hook])
        #for cache in complete_cache:
        #    mean_cache_for_hook += cache[hook]
        mean_cache[hook] = complete_cache[hook] / len(correct_queries_with_d_id)

    del complete_cache

    def residual_stream_patching_mean_in_hook(
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
        position: int
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Each HookPoint has a name attribute giving the name of the hook.
        clean_resid_pre = mean_cache[hook.name]
        resid_pre[0, position, :] = clean_resid_pre[position, :]
        return resid_pre

    def run_with_hooks(hooks):

        id_to_results = {}
        for q_id, query, correct_doc in tqdm(queries_with_d_id):
            input_tokens = get_tokens(query, pad=max_q_len+1) # query is irrelevant because we replace every input with the mean of all queries
            patched_logits = model.run_with_hooks(input_tokens, decoder_input=decoder_input, fwd_hooks=hooks)
            topk_out = patched_logits.topk(100, dim=-1)

            id_to_results[q_id] = {
                    'results': topk_out.indices[0,0].tolist(),
                    'scores': topk_out.values[0,0].tolist()
            }
        return id_to_results

    def eval(id_to_results):
    
        run = {
            "query": [],
            "q0": [],
            "docid": [],
            "rank": [], 
            "score": [],
            "system": []
        }
        relevant_for_query = defaultdict(list)
        for i, q_id in enumerate(qrels['query']):
            relevant_for_query[q_id].append(qrels['docid'][i])
        hits_1 = 0
        r_5_per_q = defaultdict(int)
        hits_10 = 0
        q_id_to_int = {}
        for i, q_id in enumerate(id_to_results):
            hits_10_doc = 0
            for rank, (doc_id_token, score) in enumerate(zip(id_to_results[q_id]['results'], id_to_results[q_id]['scores'])):
                doc_id = tokenizer.decode(doc_id_token)
                run["query"].append(i)
                q_id_to_int[q_id] = i
                run["docid"].append(doc_id)
                run["q0"].append('q0')
                run["rank"].append(rank)
                run["score"].append(score)
                run["system"].append(checkpoint)
                if doc_id in relevant_for_query[q_id]:
                    if rank == 0:
                        hits_1 += 1
                    if rank < 5:
                        r_5_per_q[q_id] += 1
                    if rank < 10:
                        hits_10_doc += 1
            if hits_10_doc > 0:
                hits_10 += 1
        hits_1 = hits_1 / len(id_to_results)
        hits_10 = hits_10 / len(id_to_results)

        for q_id in r_5_per_q:
            r_5_per_q[q_id] = r_5_per_q[q_id] / len(relevant_for_query[q_id])
                
        r_5 = sum([r_5_per_q[key] for key in r_5_per_q]) / len(id_to_results)
        
        max_id = i + 1
        new_qrel_queries = []
        for q_id in qrels['query']:
            if q_id in q_id_to_int:
                new_qrel_queries.append(q_id_to_int[q_id])
            else:
                new_qrel_queries.append(max_id)
                max_id += 1
        qrels['query'] = new_qrel_queries
        for key in run:
            print(run[key][:2])
        #results = trec_eval.compute(references=[qrels], predictions=[run])
        results = {}
        results['hits@1'] = hits_1
        results['hits@10'] = hits_10
        results['recall@5'] = r_5

        return results
                
            

   
       # these are the hooks for the final configuration!

    hooks = []

    mean_hook_fn = partial(residual_stream_patching_mean_in_hook, position=0)

    # our best combination

    for layer in range(7):
        hooks.append((f'decoder.{layer}.hook_attn_out', removal_patching_hook))
        hooks.append((f'decoder.{layer}.hook_cross_attn_out', removal_patching_hook))
        hooks.append((f'decoder.{layer}.hook_mlp_out', mean_hook_fn))

    for layer in range(18, 24):
        hooks.append((f'decoder.{layer}.hook_attn_out', removal_patching_hook))

    for layer in range(7, 18):
        hooks.append((f'decoder.{layer}.hook_attn_out', removal_patching_hook))
        hooks.append((f'decoder.{layer}.hook_mlp_out', mean_hook_fn))
    
    predictions = run_with_hooks(hooks)

    results = eval(predictions)
    results['config'] = '0-7 remove attn, remove cr-attn, mlp mean, 7 - 17 remove attn, mean mlp, 18 - 23 remove attn'


    out_path = args.output_path.replace('[checkpoint]', args.checkpoint)
    makedirs(out_path, exist_ok=True)

    store_results(results, out_path=f'{out_path}/patching_results_best_circuit.json')