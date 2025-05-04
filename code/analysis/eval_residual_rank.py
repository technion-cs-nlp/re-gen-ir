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

parser = ArgumentParser(prog='Run Evaluations on the residual stream, calculate the rank, and inspect neurons')
parser.add_argument('-c', '--checkpoint', help='checkpoint for the model to evaluate')
parser.add_argument('--val-queries', help='path for the validation queries')
parser.add_argument('--correct-valid', type=str, help='path to the correct_valid.json file', default='[checkpoint]/correct_valid.json')
parser.add_argument('-o', '--output-path', type=str, help='path to which the output files are written', default='[checkpoint]')


def get_tokens(text, pad=None):
    if not pad:
        return tokenizer(text, return_tensors='pt')['input_ids']
    else:
        return tokenizer(text, return_tensors='pt', padding='max_length', max_length=pad)['input_ids']

def tokens_to_str(tokens):
    return [tokenizer.decode(d) for d in tokens[0]]

def decoder_input():
    return torch.tensor([[0]])

def get_neuron_results_decoder(layer, cache):
    neuron_acts = cache[f'decoder.{layer}.mlp.hook_post']
    W_out = model.decoder[layer].mlp.W_out
    return neuron_acts[..., None] * W_out

def process_resid(resid, model):
    decoder_resid = model.decoder_final_ln(resid)

    if model.cfg.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        decoder_resid *= model.cfg.d_model**-0.5

    return decoder_resid

def logit_lens_decoder(resid, model):
    decoder_resid = process_resid(resid, model=model)
    logits = model.unembed(decoder_resid)

    return logits


def get_ratio(comp, i=None):
    if i == None:
        return (torch.linalg.norm(comp) / torch.linalg.norm(resid_delta)).item()
    else:
        return (torch.linalg.norm(comp) / torch.linalg.norm(resid_delta[i])).item()

def get_length(comp, i=None):
    return torch.linalg.norm(comp).item()

def get_cos(comp, i=None):
    if i == None:
        return torch.cosine_similarity(comp, resid_post).item()
    else:
        return torch.cosine_similarity(comp, resid_post[i], dim=-1).item()

def get_ranks(tensor):
    return torch.argsort(torch.argsort(tensor, descending=True))

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
    valid_queries = json.load(open(args.val_queries))

    q_id_to_q = {q['id']:q['query'] for q in valid_queries}
    correct_queries_with_d_id = [(q, q_id_to_q[q], d) for d,q in correct_valid]

    out_dir = args.output_path.replace('[checkpoint]', args.checkpoint)
    makedirs(out_dir, exist_ok=True)


    # 0. Validate correct_valid data. The base model should get all queries correct!

    all_dec_resid_results = [] 
    all_enc_resid_results = [] 
    all_ranks_correct_doc = []

    all_query_similarities = {}

    activated_neurons = {}

    for i, (q_id, query, correct) in tqdm(enumerate(correct_queries_with_d_id), total=len(correct_queries_with_d_id)):

        if i % 100 == 0:
            json.dump(all_dec_resid_results, open(f'{out_dir}/all_dec_resid_results_{i}.json', 'w'))
            json.dump(all_ranks_correct_doc, open(f'{out_dir}/all_ranks_correct_doc_{i}.json', 'w'))

        input_tokens = get_tokens(query)
        t5_logits, t5_cache = model.run_with_cache(input_tokens, decoder_input(), remove_batch_dim=True)

        all_residual_norms = []
        for layer in range(model.cfg.n_layers):

            # 1.a. Length, Angle Resid Plot for Decoder

            resid_pre = t5_cache[f'decoder.{layer}.hook_resid_pre']
            attn_out = t5_cache[f'decoder.{layer}.hook_attn_out']
            cross_attn_out = t5_cache[f'decoder.{layer}.hook_cross_attn_out']
            mlp_out = t5_cache[f'decoder.{layer}.hook_mlp_out']
            resid_post = t5_cache[f'decoder.{layer}.hook_resid_post']
            
            resid_delta = resid_post - resid_pre

            func_map = {
                'Ratio to Delta of Residual': get_ratio,
                'Absolute Value': get_length,
                'Cosine Similarity': get_cos
            }
            op_map = {
                'Self-Attn.': attn_out,
                'Cross-Attn.': cross_attn_out,
                'MLP': mlp_out,
                'Residual In': resid_pre,
                'Residual Out':resid_post
            }
            results_obj = {
                'query': q_id
            }

            for func_key in func_map:
                func = func_map[func_key]
                
                for op_key in op_map:
                    results_obj = {
                    'query': q_id,
                    'func': func_key,
                    'layer': layer,
                    'component': op_key,
                    'value': func(op_map[op_key])
                    }
                    all_dec_resid_results.append(results_obj)

            op_map = {
            'Self-Attn.': attn_out,
            'MLP': mlp_out,
            'Residual In': resid_pre,
            'Residual Out':resid_post
            }
            

            # 2. Apply logitlens to plot ranks  
            
            correct_doc_id = tokenizer.convert_tokens_to_ids(correct)

            def get_ranks_of_correct_doc(cache, component_name, layer, correct_doc_id):
                resid = cache[f'decoder.{layer}.hook_{component_name}']
                logits = logit_lens_decoder(resid, model=model).cpu()
                rank_correct = get_ranks(logits)[0][correct_doc_id].item()
                return rank_correct


            component_map = {
                'Residual Before Cr.Attn': 'resid_mid',
                'Residual After Cr.Attn': 'resid_mid_cross',
                'Cross-Attn. Output': 'cross_attn_out',
                'MLP Output': 'mlp_out',
                'Residual After Layer': 'resid_post'
            }

            for com_key in component_map:
                component = component_map[com_key]
                rank_correct = get_ranks_of_correct_doc(t5_cache, component, layer, correct_doc_id)
                data_obj = {
                    'query': q_id,
                    'layer': layer,
                    'component': com_key,
                    'value': rank_correct
                }
                all_ranks_correct_doc.append(data_obj)


            resid = t5_cache[f'decoder.{layer}.hook_resid_post']
            logits = logit_lens_decoder(resid, model=model).cpu()
            all_ranks = get_ranks(logits.flatten())
            ranks_doc_id_tokens = all_ranks[first_added_doc_id:]
            ranks_non_doc_tokens = all_ranks[:first_added_doc_id]

            rank_map ={
            'first_doc': min(ranks_doc_id_tokens).item(),
            'last_doc': max(ranks_doc_id_tokens).item(),
            'Mean Doc-IDs': (sum(ranks_doc_id_tokens)/len(ranks_doc_id_tokens)).item(),
            'first_non_doc': min(ranks_non_doc_tokens).item(),
            'last_non_doc': max(ranks_non_doc_tokens).item(),
            'Mean Non-Doc-IDs': torch.median(ranks_non_doc_tokens).item(),
            }

            for rank_key in rank_map:
                data_obj = {
                        'query': q_id,
                        'layer': layer,
                        'component': rank_key,
                        'value': rank_map[rank_key]
                }
                all_ranks_correct_doc.append(data_obj)
            


    df = pd.DataFrame(all_dec_resid_results)
    df.to_csv(out_dir+'/all_dec_resid_results.csv')

    df = pd.DataFrame(all_ranks_correct_doc)
    df.to_csv(out_dir + '/all_ranks_correct_doc.csv')

