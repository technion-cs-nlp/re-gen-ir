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
from os import makedirs


import torch 
torch.set_grad_enabled(False)

parser = ArgumentParser(prog='Run patching, single component per layer only')
parser.add_argument('-c', '--checkpoint', help='checkpoint for the model to evaluate')
parser.add_argument('--val-queries', help='path for the validation queries')
parser.add_argument('--correct-valid', type=str, help='path to the correct_valid.json file', default='[checkpoint]/correct_valid.json')
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
    
    correct_valid_path = args.correct_valid.replace('[checkpoint]', checkpoint)
    correct_valid = json.load(open(correct_valid_path))
    valid_queries = json.load(open(args.val_queries))

    q_id_to_q = {q['id']:q['query'] for q in valid_queries}
    correct_queries_with_d_id = [(q, q_id_to_q[q], d) for d,q in correct_valid]

    lengths = []
    for q_id, query, doc in correct_queries_with_d_id:
        lengths.append(len(tokenizer.tokenize(query)))

    max_q_len = max(lengths)

    complete_cache = {}
    mean_cache = {}
    for q_id, query, doc in tqdm(correct_queries_with_d_id):
        input_tokens = get_tokens(query, pad=max_q_len+1)
        t5_logits, t5_cache = model.run_with_cache(input_tokens, decoder_input, remove_batch_dim=True)
        for key in t5_cache:
            if key not in complete_cache:
                complete_cache[key] = t5_cache[key]
            else:
                complete_cache[key] += t5_cache[key]

    mean_cache = {}
    for hook in complete_cache:

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
    


    stages = {
        0: [0,1,2,3,4,5,6],
        1: [7,8,9,10,11,12,13,14,15,16,17],
        2: [18,19,20,21,22,23]
    }


    hooks = []

    mean_hook_fn = partial(residual_stream_patching_mean_in_hook, position=0)

    no_stages = 3
    components = ['mlp', 'cross_attn', 'attn']
    mode = ['mean', 'zero']


    single_combinations = [{'component': c,
        'stage': s,
        'mode': m} for m in mode for c in components for s in range(no_stages) ]
    
    all_combis_correct_ranks = {}

    for combi in single_combinations:
        hooks = []
        hook = mean_hook_fn if combi['mode'] == 'mean' else removal_patching_hook
        comp = combi['component']
        layers = stages[combi['stage']]
        for layer in layers:
            hooks.append((f'decoder.{layer}.hook_{comp}_out', hook))

        correct_top_0 = []
        all_ranks_correct_doc = []

        for q_id, query, correct_doc in tqdm(correct_queries_with_d_id):
            input_tokens = get_tokens(query, pad=max_q_len+1)
            patched_logits = model.run_with_hooks(input_tokens, decoder_input=decoder_input, fwd_hooks=hooks)
            max_tok = patched_logits.argmax(dim=-1)
            correct_doc_id = tokenizer.convert_tokens_to_ids(correct_doc)
            rank_correct_doc = get_ranks(patched_logits)[0][0][correct_doc_id]
            all_ranks_correct_doc.append(rank_correct_doc.item())

        combi['!=0'] = len([r for r in all_ranks_correct_doc if r != 0]) / len(all_ranks_correct_doc)
        combi['>5'] = len([r for r in all_ranks_correct_doc if r > 5]) / len(all_ranks_correct_doc)
        combi['>10'] = len([r for r in all_ranks_correct_doc if r > 10]) / len(all_ranks_correct_doc)
        combi['>100'] = len([r for r in all_ranks_correct_doc if r > 100]) / len(all_ranks_correct_doc)

        key = f"{combi['stage']}-{combi['component']}-{combi['mode']}"
        all_combis_correct_ranks[key] = all_ranks_correct_doc

   
    out_path = args.output_path.replace('[checkpoint]', args.checkpoint)
    makedirs(out_path, exist_ok=True)


    df = pd.DataFrame(single_combinations)
    df.to_csv(out_path+'/patching_results_single_combis.csv')

    json.dump(all_combis_correct_ranks, open(f'{out_path}/ranks_correct_queries_patching_single_combis.json', 'w'))
