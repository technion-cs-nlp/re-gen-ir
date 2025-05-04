from transformer_lens import HookedEncoderDecoder
import transformer_lens.utils as utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
from argparse import ArgumentParser
from tqdm import tqdm
from os import makedirs

import json
import pandas as pd
import torch 
import einops


torch.set_grad_enabled(False)

parser = ArgumentParser(prog='Find from where cross-attention reads')
parser.add_argument('-c', '--checkpoint', help='checkpoint for the model to evaluate')
parser.add_argument('--val-queries', help='path for the validation queries')
parser.add_argument('--sim-type', help='cosine or dot')
parser.add_argument('--correct-valid', type=str, help='path to the correct_valid.json file', default='[checkpoint]/correct_valid.json')
parser.add_argument('-o', '--output-path', type=str, help='path to which the output files are written', default='[checkpoint]')



def get_tokens(text, pad=None):
    if not pad:
        return tokenizer(text, return_tensors='pt')['input_ids']
    else:
        return tokenizer(text, return_tensors='pt', padding='max_length', max_length=pad)['input_ids']


def decoder_input():
    return torch.tensor([[0]])


# https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/components/abstract_attention.py
def get_attn_result(model, cache, layer, component='cross_attn'):
    full_name = f'decoder.{layer}.{component}.hook_z'
    z = cache[full_name]
    z = einops.rearrange(
        z, "pos head_index d_head -> pos head_index d_head 1"
    )

    if component == 'attn':
        w = model.decoder[layer].attn.W_O
    elif component == 'cross_attn':
        w = model.decoder[layer].cross_attn.W_O
    else:
        print(component, 'is not supported.')
        return

    w = einops.rearrange(
                    w,
                    "head_index d_head d_model -> 1 head_index d_head d_model",
                )
    
    result = (z * w).sum(-2)
    return result

def similarity(vec1, vec2, dim, sim_type='cosine'):
    if sim_type=='cosine':
        return torch.cosine_similarity(vec1, vec2, dim=dim)
    else:
        return (vec1 * vec2).sum(dim=dim)

def get_per_head_similarity(vector, cache, layer, component, model, sim_type):
    component = component[5:-4]
    per_head_vector = get_attn_result(model=model, cache=cache, layer=layer, component=component)
    per_head_sim = similarity(vector, per_head_vector, dim=-1, sim_type=sim_type)
    return per_head_sim[0]


def search_for_components(vector, cache, layer, model, component='hook_mlp_out', sim_type='cosine'):
    similarities = {} # component -> % of similarity
    components_per_layer = ['hook_resid_pre', 'hook_attn_out', 'hook_cross_attn_out', 'hook_resid_mid_cross', 'hook_mlp_out', 'hook_resid_post']
    for prev_layer in range(layer):
        for prev_component in components_per_layer:
            full_name = f'decoder.{prev_layer}.{prev_component}'
            comp_vec = cache[full_name][0]
            sim = similarity(vector, comp_vec, dim=-1, sim_type=sim_type).item()
            similarities[full_name] = sim
            if 'attn' in prev_component:
                key_name = f'decoder.{prev_layer}.{prev_component[5:-4]}'
                per_head_sim = get_per_head_similarity(vector, cache, prev_layer, prev_component, model, sim_type)
                for head_idx in range(len(per_head_sim)):
                    similarities[f'{key_name}.head{head_idx}'] = per_head_sim[head_idx].item()
    current_comp_index = components_per_layer.index(component)
    for prev_component in components_per_layer[:current_comp_index]:
        full_name = f'decoder.{layer}.{prev_component}'
        comp_vec = cache[full_name][0]
        sim = similarity(vector, comp_vec, dim=-1, sim_type=sim_type).item()
        similarities[full_name] = sim
    return similarities

def search_for_components_from_attn(wq, vector, cache, layer, model, component='hook_mlp_out', sim_type='cosine'):
    similarities = {} # component -> % of similarity
    components_per_layer = ['hook_resid_pre', 'hook_attn_out', 'hook_cross_attn_out', 'hook_resid_mid_cross', 'hook_mlp_out', 'hook_resid_post']

    for prev_layer in range(layer):
        for prev_component in components_per_layer:
            full_name = f'decoder.{prev_layer}.{prev_component}'
            comp_vec = torch.matmul(cache[full_name][0], wq)
            sim = similarity(vector, comp_vec, dim=-1, sim_type=sim_type).item()
            similarities[full_name] = sim
            #if 'attn' in prev_component:
            #    key_name = f'decoder.{prev_layer}.{prev_component[5:-4]}'
            #    per_head_sim = get_per_head_similarity(vector, cache, prev_layer, prev_component, model, sim_type)
            #    for head_idx in range(len(per_head_sim)):
            #        similarities[f'{key_name}.head{head_idx}'] = per_head_sim[head_idx].item()
    current_comp_index = components_per_layer.index(component)
    for prev_component in components_per_layer[:current_comp_index]:
        full_name = f'decoder.{layer}.{prev_component}'
        comp_vec = torch.matmul(cache[full_name][0], wq)
        sim = similarity(vector, comp_vec, dim=-1, sim_type=sim_type).item()
        similarities[full_name] = sim
    return similarities

if __name__ == '__main__':

    args = parser.parse_args()


    # Load Model into Transformer Lens
    print('Loading model...')
    checkpoint = args.checkpoint
    OFFICIAL_MODEL_NAMES.append(checkpoint)

    hf_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    device = utils.get_device()
    model = HookedEncoderDecoder.from_pretrained(checkpoint, hf_model=hf_model, device=device)

    tokenizer_t5 = AutoTokenizer.from_pretrained('google-t5/t5-large')


    first_added_doc_id = len(tokenizer_t5)
    last_added_doc_id = len(tokenizer_t5) + (len(tokenizer) - len(tokenizer_t5))
    del tokenizer_t5

    # Load eval data
    print('Loading data...')
    correct_valid_path = args.correct_valid.replace('[checkpoint]', checkpoint)
    correct_valid = json.load(open(correct_valid_path))
    valid_queries = json.load(open(args.val_queries)) #val_queries-25-6959.json

    q_id_to_q = {q['id']:q['query'] for q in valid_queries}
    correct_queries_with_d_id = [(q, q_id_to_q[q], d) for d,q in correct_valid]

    # 0. Validate correct_valid data. The base model should get all queries correct!

    out_dir = args.output_path.replace('[checkpoint]', args.checkpoint)
    makedirs(out_dir, exist_ok=True)

    all_query_similarities = []

    activated_neurons = {}
    for i, (q_id, query, correct) in tqdm(enumerate(correct_queries_with_d_id), total=len(correct_queries_with_d_id)):
        if i % 50 == 0:
            json.dump(all_query_similarities, open(out_dir + f'/all_queries_crattn_similarities_{args.sim_type}_{i}.json', 'w'))
            all_query_similarities = []
        input_tokens = get_tokens(query)
        t5_logits, t5_cache = model.run_with_cache(input_tokens, decoder_input(), remove_batch_dim=True)

        all_residual_norms = []
        for layer in range(model.cfg.n_layers):

            # 1.a. Length, Angle Resid Plot for Decoder

            # 3. Determine which neurons get activated in each layer (only later layers?)
            best_enc_positions = torch.argmax(t5_cache[f'decoder.{layer}.cross_attn.hook_attn_scores'], dim=-1).flatten()
            key_matrix = t5_cache[f'decoder.{layer}.cross_attn.hook_k']
            wq = model.decoder[layer].cross_attn.W_Q
            for head, position in enumerate(best_enc_positions):
                key_vector = key_matrix[position, head]
                similarities = search_for_components_from_attn(wq[head], key_vector, model=model, cache=t5_cache, layer=layer, component='hook_cross_attn_out', sim_type=args.sim_type)
                data_obj ={
                        'sim_type': args.sim_type,
                        'value': similarities,
                        'layer':layer,
                        'head': head,
                        'q_id':q_id,
                        'enc_position': position.item()
                }
                all_query_similarities.append(data_obj)




    # Write out all data!
    
    json.dump(all_query_similarities, open(out_dir + f'/all_queries_crattn_similarities_{args.sim_type}_{i}.json', 'w'))
    
