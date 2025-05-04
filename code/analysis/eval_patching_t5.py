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
parser.add_argument('--val-queries', help='path for the validation queries')
parser.add_argument('--qrels', help='path for the validation qrels')
parser.add_argument('--t5', help="which version of T5 to use, default=t5-large", default="t5-large")
parser.add_argument('-o', '--output-path', type=str, help='path to which the output files are written', default='[checkpoint]')




def removal_patching_hook(
    emb: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
) -> Float[torch.Tensor, "batch pos d_model"]:
    # Each HookPoint has a name attribute giving the name of the hook.
    
    emb[:, :] = torch.zeros_like(emb)
    return emb


def get_ranks(tensor):
    return torch.argsort(torch.argsort(tensor, descending=True))

def get_tokens(text, tokenizer, pad=None):
    if not pad:
        return tokenizer(text, return_tensors='pt')['input_ids']
    else:
        return tokenizer(text, return_tensors='pt', padding='max_length', max_length=pad)['input_ids']

def tokens_to_str(tokens):
    return [tokenizer.decode(d) for d in tokens[0]]

decoder_input = torch.tensor([[0]])


# based on https://stackoverflow.com/a/57915246
def store_results(results, file_path):

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    json.dump(results, open(file_path, 'w'), cls=NpEncoder)


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
    qrels = json.load(open(args.qrels))

    q_id_to_q = {q['id']:q['query'] for q in valid_queries}
    correct_queries_with_d_id = [(q, q_id_to_q[q], d) for d,q in correct_valid]
    queries_with_d_id = [(q['id'], q['query'], "") for q in valid_queries]

    stages = {
        0: [0,1,2,3,4,5,6],
        1: [7,8,9,10,11,12,13,14,15,16,17],
        2: [18,19,20,21,22,23]
    }    

    device = utils.get_device()

    model_t5 = HookedEncoderDecoder.from_pretrained(args.t5, device=device)
    tokenizer_t5 = AutoTokenizer.from_pretrained('google-t5/'+args.t5)

    def run_with_hooks(hooks):

        id_to_results = {}
        for q_id, query, correct_doc in tqdm(queries_with_d_id):

            input_tokens = get_tokens(query, tokenizer_t5)
            _, t5_cache = model_t5.run_with_cache(input_tokens, decoder_input )

            def t5_patching_hook(
                emb: Float[torch.Tensor, "batch pos d_model"],
                hook: HookPoint,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                # Each HookPoint has a name attribute giving the name of the hook.
                t5_emb = t5_cache[hook.name]
                emb[:, :] = t5_emb[:, :]
                return emb
            

            hooks = []
            for layer in range(7):
                hooks.append((f'decoder.{layer}.hook_attn_out', t5_patching_hook))
                hooks.append((f'decoder.{layer}.hook_cross_attn_out', t5_patching_hook))
                hooks.append((f'decoder.{layer}.hook_mlp_out', t5_patching_hook))

            for layer in range(18, 24):
                hooks.append((f'decoder.{layer}.hook_attn_out', t5_patching_hook))

            for layer in range(7, 18):
                hooks.append((f'decoder.{layer}.hook_attn_out', t5_patching_hook))
                hooks.append((f'decoder.{layer}.hook_mlp_out', t5_patching_hook))
                hooks.append((f'decoder.{layer}.hook_cross_attn_out', t5_patching_hook))
            



            input_tokens = get_tokens(query, tokenizer) # query is irrelevant because we replace every input with the mean of all queries
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
        results = trec_eval.compute(references=[qrels], predictions=[run])
        results['hits@1'] = hits_1
        results['hits@10'] = hits_10
        results['recall@5'] = r_5

        return results
                
            

    no_stages = 3
    components = ['mlp', 'cross_attn', 'attn', 'encoder.23.hook_resid_post']
    mode = ['t5']


    single_combinations = [{'component': c,
        'stage': s,
        'mode': m} for m in mode for c in components for s in range(no_stages) ]
    
    all_combis_correct_ranks = {}

    for combi in single_combinations:
        

        correct_top_0 = []
        all_ranks_correct_doc = []

        for q_id, query, correct_doc in tqdm(correct_queries_with_d_id):
            input_tokens = get_tokens(query, tokenizer_t5)

            _, t5_cache = model_t5.run_with_cache(input_tokens, decoder_input)
            def t5_patching_hook(
                emb: Float[torch.Tensor, "batch pos d_model"],
                hook: HookPoint,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                # Each HookPoint has a name attribute giving the name of the hook.
                t5_emb = t5_cache[hook.name]
                emb[:, :] = t5_emb[:, :]
                return emb
            
            hooks = []
            hook = t5_patching_hook
            comp = combi['component']
            stage = combi['stage']
            layers = stages[combi['stage']]
            if 'encoder' in comp and stage == 0:
                hooks.append((comp, hook))
            elif 'encoder' in comp and stage > 0:
                continue
            else:
                for layer in layers:
                    hooks.append((f'decoder.{layer}.hook_{comp}_out', hook))

            input_tokens = get_tokens(query, tokenizer) # query is irrelevant because we replace every input with the mean of all queries
            patched_logits = model.run_with_hooks(input_tokens, decoder_input=decoder_input, fwd_hooks=hooks)
            max_tok = patched_logits.argmax(dim=-1)
            correct_doc_id = tokenizer.convert_tokens_to_ids(correct_doc)
            print(patched_logits.shape, get_ranks(patched_logits).shape)
            rank_correct_doc = get_ranks(patched_logits)[0][0][correct_doc_id]
            all_ranks_correct_doc.append(rank_correct_doc.item())
            if tokenizer.decode(max_tok[0][0]) == correct_doc:
                correct_top_0.append((query, correct_doc))

        if  len(all_ranks_correct_doc) != 0:
            combi['!=0'] = len([r for r in all_ranks_correct_doc if r != 0]) / len(all_ranks_correct_doc)
            combi['>5'] = len([r for r in all_ranks_correct_doc if r > 5]) / len(all_ranks_correct_doc)
            combi['>10'] = len([r for r in all_ranks_correct_doc if r > 10]) / len(all_ranks_correct_doc)
            combi['>100'] = len([r for r in all_ranks_correct_doc if r > 100]) / len(all_ranks_correct_doc)
        else:
            combi['!=0'] = 0
            combi['>5'] = 0
            combi['>10'] = 0
            combi['>100'] = 0

        key = f"{combi['stage']}-{combi['component']}-{combi['mode']}"
        all_combis_correct_ranks[key] = all_ranks_correct_doc


    out_path = args.output_path.replace('[checkpoint]', args.checkpoint)
    makedirs(out_path, exist_ok=True)


    df = pd.DataFrame(single_combinations)
    df.to_csv(out_path+'/patching_results_t5_single_combis.csv')

    json.dump(all_combis_correct_ranks, open(f'{out_path}/ranks_correct_queries_patching_t5_single_combis.json', 'w'))
   

    
    
    predictions = run_with_hooks(hooks)

    results = eval(predictions)
    results['config'] = '0-6 attn, cr, mlp = T5, 7 - 17 attn, cr, mlp = T5, 18 - 23 only attn = t5'

    #json.dump(results, open(out_path+'/patching_results_t5_all_and_attn.json', 'w'))
    store_results(results, out_path+'/patching_results_t5_all_and_attn.json')


    
