try:
    import comet_ml
    comet_installed = True
except ModuleNotFoundError:
    print("comet_ml is not installed.")
    comet_installed = False

from torch.utils.data import DataLoader
from evaluate import load
from argparse import ArgumentParser
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, TextGenerationPipeline
from collections import defaultdict
from utils.utils import *

import torch
import json
import numpy as np


parser = ArgumentParser(prog='Team Evaluation', description='Evaluate a trained model')
parser.add_argument('-d', '--data', help='path of the pre-processed data (including train, eval/dev)', required=True)
parser.add_argument('-t', '--model-type', help='are we training a encoder-decoder or only a decoder type model', 
                    required=False, default='encoder-decoder', choices=['encoder-decoder', 'decoder'])
parser.add_argument('-c', '--use_comet', help='use coment_ml to track experiments or not', required=False, default=False, action='store_true')
parser.add_argument('-r', '--ratio', help='the ratio of indexing to retrieval examples', default=1)
parser.add_argument('--doc-ids', help='the method of creating doc ids', default='atomic', choices=['naive', 'semantic', 'atomic'])
parser.add_argument('-e', '--experiment-dir', help='the output directory of the model training, i.e., where to load the model from', default='experiment')
parser.add_argument('--test-batch-size', help='batch size used for evaluation', default=2)
parser.add_argument('--seed', help='seed for the dataset', default=0)


trec_eval = load("trec_eval")

def get_predictions_encoder_decoder(args, test_ds, model, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data_loader = DataLoader(test_ds, batch_size=args.test_batch_size)
    id_to_results = {}

    if args.doc_ids == 'atomic':

        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            decoder_input_ids = torch.zeros(input_ids.shape[0],1).long().to(device)
            output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            topk_out = output['logits'].topk(100, 2)
            for i, (q_id, input_ids) in enumerate(zip(batch['query_id'], batch['input_ids'])):
                q_id = q_id.item() if type(q_id) != str else q_id
                id_to_results[q_id] = {
                    'results': topk_out.indices[i,0].tolist(),
                    'scores': topk_out.values[i,0].tolist(),
                    'input_ids': input_ids
                }
    else:
        eval_pipeline = pipeline(model=model, tokenizer=tokenizer, task='translation', device=device)
        for i, examples in enumerate(test_data_loader):
            input_ids = examples['input_ids']
            input_text = tokenizer.batch_decode(input_ids)
            output = eval_pipeline(input_text, num_beams=10, num_return_sequences=10)
            for j, q_id in enumerate(examples['query_id']):
                id_to_results[q_id] = {
                    'results': [r['translation_text'].replace(' ', '_') for r in output[j]],
                    'scores': [len(output[j]) - rank for rank, _ in enumerate(output[j])],
                    'input_ids': input_ids[j]
                }
    return id_to_results

def get_predictions_decoder(args, test_ds, model, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data_loader = DataLoader(test_ds, batch_size=args.per_device_eval_batch_size)
    id_to_results = {}

    if args.doc_ids == 'atomic':
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            output = model(input_ids=input_ids)
            topk_out = output['logits'].topk(100, 2)
            for i, (q_id, input_ids) in enumerate(zip(batch['query_id'], batch['input_ids'])):
                mask = list([t[i] for t in batch['attention_mask']])
                try:
                    last_token = mask.index(0) - 1
                except:
                    last_token = -1
                q_id = q_id.item() if type(q_id) != str else q_id
                id_to_results[q_id] = {
                    'results': topk_out.indices[i, last_token].tolist(),
                    'scores': topk_out.values[i, last_token].tolist(),
                    'input_ids': input_ids
                }
    else:
        eval_pipeline = TextGenerationPipeline(model, tokenizer)
        for i, examples in enumerate(test_data_loader):
            input_ids = examples['input_ids']
            input_text = tokenizer.batch_decode(input_ids)
            output = eval_pipeline(input_text, num_beams=10, num_return_sequences=10)
            for j, q_id in enumerate(examples['query_id']):
                id_to_results[q_id] = {
                    'results': [r['generated_text'].replace(' ', '_') for r in output[j]],
                    'scores': [len(output[j]) - rank for rank, _ in enumerate(output[j])],
                    'input_ids': input_ids[j]
                }

    return id_to_results

def evaluate(args, test_ds, qrels, model, tokenizer, experiment_dir, accelerator=None, data_prefix='test'):

    test_ds.set_format("torch") 
    id_to_results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        
        if args.model_type == 'encoder-decoder':
            id_to_results = get_predictions_encoder_decoder(args, test_ds, model, tokenizer)
        else:
            id_to_results = get_predictions_decoder(args, test_ds, model, tokenizer)
                
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
    hits_10 = 0
    r_5_per_q = defaultdict(int)
    
    q_id_to_int = {}
    correct = []
    for i, q_id in enumerate(id_to_results):
        hits_10_doc = 0
        for rank, (doc_id_token, score) in enumerate(zip(id_to_results[q_id]['results'], id_to_results[q_id]['scores'])):
            if args.doc_ids == 'atomic':
                doc_id = tokenizer.decode(doc_id_token)
            else:
                doc_id = doc_id_token
            run["query"].append(i)
            q_id_to_int[q_id] = i
            run["docid"].append(doc_id)
            run["q0"].append('q0')
            run["rank"].append(rank)
            run["score"].append(score)
            run["system"].append(experiment_dir)
            if doc_id in relevant_for_query[q_id]:
                if rank == 0:
                    hits_1 += 1
                    correct.append((doc_id, q_id))
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

   
    pprint_results(results)

    return results, correct

def pprint_results(results):
    max_key_length = max(len(str(key)) for key in results) + 4

    for key in results:
        if isinstance(results[key], float):
            print(f'{key.ljust(max_key_length)} {results[key]:.4}')
        else:
            print(f'{key.ljust(max_key_length)} {results[key]}')

# based on https://stackoverflow.com/a/57915246
def store_results(results, results_dir, data_prefix, data_dir):

    results['data_dir'] = data_dir

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    json.dump(results, open(f'{results_dir}/{data_prefix}_results.json', 'w'), cls=NpEncoder)





def main():
    args = parser.parse_args()
    print("Loading dataset...")
    data_dir = get_dataset_dir(args)
    ds = load_from_disk(data_dir)
    qrels_valid = json.load(open(f"{data_dir}/valid_qrels.json"))
    qrels_test = json.load(open(f"{data_dir}/test_qrels.json"))

    print("Loading model and tokenizer...")

    model_dir = args.experiment_dir
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print("Eval on Validation data:")
    results, correct = evaluate(args, ds['complete_valid'], qrels_valid, model, tokenizer, model_dir, data_prefix="valid")
    store_results(results, model_dir, "retrieval_valid", data_dir=data_dir)
    json.dump(correct, open(f'{model_dir}/correct_valid.json', 'w'))
    print("Eval on Test data:")
    results, correct = evaluate(args, ds['test'], qrels_test, model, tokenizer, model_dir)
    json.dump(correct, open(f'{model_dir}/correct_test.json', 'w'))
    store_results(results, model_dir, "retrieval_test", data_dir=data_dir)

        



if __name__ == '__main__':
    main()
