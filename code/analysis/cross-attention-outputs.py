from transformer_lens import HookedEncoderDecoder
import transformer_lens.utils as utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES


from argparse import ArgumentParser
from tqdm import tqdm
import json
from collections import defaultdict
from os import makedirs


import torch 
torch.set_grad_enabled(False)

parser = ArgumentParser(prog='Ratio of doc-ids vs. non-doc-ids promoted in cross-attention output')
parser.add_argument('-c', '--checkpoint', help='checkpoint for the model to evaluate')
parser.add_argument('--correct-valid', type=str, help='path to the correct_valid.json file', default='[checkpoint]/correct_valid.json')
parser.add_argument('--val-queries', help='path for the validation queries')
parser.add_argument('-o', '--output-path', type=str, help='path to which the output files are written', default='[checkpoint]')



def get_tokens(text, pad=None):
    if not pad:
        return tokenizer(text, return_tensors='pt')['input_ids']
    else:
        return tokenizer(text, return_tensors='pt', padding='max_length', max_length=pad)['input_ids']

decoder_input = torch.tensor([[0]])


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

    q_id_to_q = {q['id']:q['query'] for q in valid_queries}
    correct_queries_with_d_id = [(q, q_id_to_q[q], d) for d,q in correct_valid]
    queries_with_d_id = [(q['id'], q['query'], "") for q in valid_queries]

    tokenizer_t5 = AutoTokenizer.from_pretrained('google-t5/t5-large')

    first_added_doc_id = len(tokenizer_t5)
    last_added_doc_id = len(tokenizer_t5) + (len(tokenizer) - len(tokenizer_t5))
    del tokenizer_t5

    out_dir = args.output_path.replace('[checkpoint]', args.checkpoint)
    makedirs(out_dir, exist_ok=True)



    per_q_averages_sII = defaultdict(list)
    per_q_averages_sIII = defaultdict(list)
    for i, (q_id, query, doc) in tqdm(enumerate(correct_queries_with_d_id), total=len(correct_queries_with_d_id)):
        input = get_tokens(query)
        _, activation_cache = model.run_with_cache(input, decoder_input=decoder_input)
        for layer in range(7,24):

            neuron_activations = activation_cache[f'decoder.{layer}.hook_cross_attn_out']
            value_logits = logit_lens_decoder(neuron_activations, model=model).cpu()
            topk_res = torch.topk(value_logits, k=1000)

            num_doc_ids = defaultdict(int)
            for i in range(1000):
                if topk_res.indices[0][0][i] >= first_added_doc_id:
                    if i < 10:
                        num_doc_ids["10"] += 1
                    if i < 100:
                        num_doc_ids["100"] += 1
                    
                    num_doc_ids["1000"] += 1
            for key in ["10", "100", "1000"]:
                if layer < 17:
                    per_q_averages_sII[key].append(num_doc_ids[key] / int(key))
                else:
                    per_q_averages_sIII[key].append(num_doc_ids[key] / int(key))

    json.dump(per_q_averages_sII, open(f'{out_dir}/per_q_averages_sII.json', 'w'))
    json.dump(per_q_averages_sIII, open(f'{out_dir}/per_q_averages_sIII.json', 'w'))

    averages = {}
    for key in per_q_averages_sII:
        averages[f"{key}-II"] = sum(per_q_averages_sII[key]) / len(per_q_averages_sII[key])
    for key in per_q_averages_sII:
        averages[f"{key}-III"] = sum(per_q_averages_sIII[key]) / len(per_q_averages_sIII[key])

    json.dump(averages, open(f"{out_dir}/cr_attn_ratios.json", "w"))
