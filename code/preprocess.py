from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict, load_dataset
from os import makedirs
from collections import defaultdict
from functools import partial

import string

from utils.utils import *

import json

parser = ArgumentParser(prog='Team Preprocessing', description='Preprocessing for GenIR')
parser.add_argument('-i', '--input', help='input path of the data', required=False)
parser.add_argument('-d', '--documents', help='the document corpus', required=True)
parser.add_argument('-t', '--train', help='queries for training', required=True)
parser.add_argument('-v', '--val', help='queries for validation', required=True)
parser.add_argument('--test', help='queries for test', required=False, default=None)
parser.add_argument('-o', '--output', help='output path of the processed data', required=True)
parser.add_argument('-m', '--model-path', help='model identifier from the huggingface hub or path', required=True)
parser.add_argument('--tokenizer', help='if you want to use another tokenizer than the model_path tokenizer, please specify it here', required=False, default=None)
parser.add_argument('-r', '--ratio', help='the ratio of indexing to retrieval examples, int for ratio, None for no increased/decreased ratio', default=1)
parser.add_argument('--max-length', help='maximum length of the input to the model (document + doc_id)', default=128, type=int)
parser.add_argument('--doc-ids', help='the method of creating doc ids', default='atomic', choices=['naive', 'semantic', 'atomic'])
parser.add_argument('--prefix', help='prefix prepended to the input string (e.g., a prompt)', default='')
parser.add_argument('--token', help='access token for gated HF models and tokenizers', default=None)
parser.add_argument('--semantic-doc-id-mapping', help='mapping from doc ids to semantic doc ids', default=None)

text_max_length = 1000
def convert_doc_id_semantic(mapping, doc_id):
    return mapping[str(doc_id)]

def create_retrieval_data(queries, doc_id_fct=convert_doc_id_token):
    retrieval_data = []
    doc_ids = [] 
    for query in queries:
        for rank, doc_id in enumerate(query['relevant_docs']):
            doc_ids.append(doc_id_fct(doc_id))
            retrieval_data.append({'text': query['query'], 'doc_id': doc_id_fct(doc_id), 'rank': rank})
            
    return retrieval_data, doc_ids

def create_test_data(queries, doc_id_fct=convert_doc_id_token):
    test_data = []
    qrels = {
        "query": [],
        "q0": [],
        "docid": [],
        "rel": []
    }
    doc_ids = []
    for query in queries:
        for rank, doc_id in enumerate(query['relevant_docs']):
            doc_ids.append(doc_id_fct(doc_id))
            qrels["query"].append(query['id'])
            qrels["docid"].append(doc_id_fct(doc_id))
            qrels["q0"].append('q0')
            qrels["rel"].append(4 - rank) # our first document is more relevant than the documents are the second rank, most relevant = 4, second document = 3
            # might need to revised for other datasets. Maybe that should be hardcoded into the dataset as it usually comes from the annotators.

        test_data.append({'text': query['query'], 'query_id': query['id']})

    return test_data, qrels, doc_ids
    
def create_index_data(documents, doc_id_fct=convert_doc_id_token):
    index_data = []
    all_doc_ids = []

    for doc in documents:
        converted_doc_ids = doc_id_fct(doc['id'])
        index_data.append({'text':doc['text'][:text_max_length], 'doc_id': converted_doc_ids})
        all_doc_ids.append(converted_doc_ids)

    return index_data, all_doc_ids

def resize_vocab(all_doc_ids, model, tokenizer):

    added_tokens = tokenizer.add_tokens(all_doc_ids)
    model.resize_token_embeddings(len(tokenizer))    
    
    return tokenizer, model

def create_dataset(index_data, retrieval_data, val_retrieval_data, complete_val_retrieval_data, test_retrieval_data=None, ratio=1):
    
    no_queries = len(retrieval_data)
    no_docs = len(index_data)
    if not (ratio == None or ratio == "None"):
        ratio = int(ratio)
        if no_docs < no_queries * ratio:
            doc_ratio = (no_queries * ratio / no_docs) - 1
            new_index = [e for e in index_data]
            while doc_ratio > 1:
                new_index = new_index + index_data
                doc_ratio -= 1
            new_length = round(doc_ratio * no_docs)
            new_index += index_data[:new_length]
            index_data = new_index
    
        elif no_docs > no_queries * ratio:
            query_ratio = (no_docs / ratio / no_queries) - 1
            new_queries = [e for e in retrieval_data]
            while query_ratio > 1:
                new_queries = new_queries + retrieval_data
                query_ratio -= 1
            new_length = round(query_ratio * no_queries)
            new_queries += retrieval_data[:new_length]
            retrieval_data = new_queries
    print("Indexing:", len(index_data), " examples, Retrieval:", len(retrieval_data), " examples.")
    train_dataset = Dataset.from_list(index_data + retrieval_data)
    val_dataset = Dataset.from_list(val_retrieval_data)
    complete_val_dataset = Dataset.from_list(complete_val_retrieval_data)

    if test_retrieval_data:
        test_dataset = Dataset.from_list(test_retrieval_data)
        ds = DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'valid': val_dataset,
            'complete_valid': complete_val_dataset})
    else:
        ds = DatasetDict({
            'train': train_dataset,
            'valid': val_dataset,
            'complete_valid': complete_val_dataset})
        
    return ds
    

def tokenize_data(ds, tokenizer, args, model_type):

    def split_input(input_ids, special_token_mask, doc_end, len_prompt):
        start_tokens = []
        document_tokens = []
        prompt_tokens = []
        doc_id_tokens = []
        end_tokens = []

        for i, (id, mask) in enumerate(zip(input_ids, special_token_mask)):
            if mask == 1:
                if len(document_tokens) == 0:
                    start_tokens.append(id)
                else:
                    end_tokens.append(id)
            else:
                if i <= doc_end:
                    document_tokens.append(id)
                elif i <= doc_end + len_prompt:
                    prompt_tokens.append(id)
                else:
                    doc_id_tokens.append(id)
        return start_tokens, document_tokens, prompt_tokens, doc_id_tokens, end_tokens

    def truncate_doc(example, target=None):
        tokenized = tokenizer(example, truncation=False, padding=False, return_special_tokens_mask=True)
        doc_ids_sep = tokenizer.encode(" Document:", add_special_tokens=False)
        # find begin of "Document:"
        # we go backwards through the document to find the beginning 
        found_all = [False for id in doc_ids_sep]
        last_found_id = -1
        for i in range(len(tokenized['input_ids'][:])-1, -1, -1):
            if False not in found_all:
                # if we found all ids of the target beginning:
                # i is first id where document begins
                break

            current_id = tokenized['input_ids'][i]
            if current_id not in doc_ids_sep:
                continue

            # check if last found id was in previous position
            if last_found_id - 1 == i or last_found_id == -1:
                last_found_id = i
            else:
                last_found_id = -1
                continue

            # check that we also found all other ids before that one
            curr_id_idx = doc_ids_sep.index(current_id)
            next_false_idx = found_all.index(False)
            if (len(doc_ids_sep) - 1 - curr_id_idx) == next_false_idx:
                found_all[next_false_idx] = True

        start_tokens, document, prompt_tokens, doc_id_tokens, end_tokens = split_input(tokenized['input_ids'], tokenized['special_tokens_mask'], i, len(doc_ids_sep))

        if target:
            if args.doc_ids == 'semantic':
                target = tokenizer(target.split('_'), is_split_into_words=True, truncation=False, add_special_tokens=False)['input_ids']
            else:
                target = tokenizer(target, truncation=False, add_special_tokens=False)['input_ids']
        else:
            target = []

        len_of_non_doc_tokens = len(start_tokens) + len(prompt_tokens) + len(target) + len(end_tokens)

        new_doc = document[:args.max_length]
        input_ids = start_tokens + new_doc + prompt_tokens + target + end_tokens
        max_len_total = args.max_length + len(start_tokens) + len(prompt_tokens) +  len(end_tokens) + 10 # 10 for a max estimate for the length of the target
        if not target:
            input_ids = input_ids + [tokenizer.pad_token_id] * (max_len_total - len(input_ids))
            mask = [1]*len(input_ids) + [0] * (max_len_total - len(input_ids))
            return {'input_ids': input_ids, 'attention_mask': mask}
        else:
            len_before_target = len(start_tokens) + len(new_doc) + len(prompt_tokens)
            targets = [-100] * len_before_target + target + end_tokens
            return {'input_ids': input_ids, 'attention_mask': [1]*len(input_ids), 'labels': targets}
        

    def preprocess_function(examples):
        if model_type == 'encoder-decoder':
            inputs = [args.prefix + example for example in examples['text']]
            if 'doc_id' in examples:
                targets = examples['doc_id']
                if args.doc_ids == 'naive':
                    model_inputs = tokenizer(inputs, text_target=targets, max_length=args.max_length, truncation=True)
                elif args.doc_ids == 'atomic':
                    model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True)
                    targets = tokenizer("", text_target=targets, add_special_tokens=False)
                    model_inputs['labels'] = targets['labels']
                elif args.doc_ids == 'semantic':
                    targets = [e.split('_') for e in targets]
                    model_inputs = {}
                    model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True)
                    targets = tokenizer([], text_target=targets, is_split_into_words=True, max_length=args.max_length, truncation=True)
                    model_inputs['labels'] = targets['labels']

            else:
                model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True, padding="max_length")
        elif model_type == 'decoder-only':
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model_inputs = defaultdict(list)
            if 'doc_id' in examples:
                inputs = [truncate_doc(f"{text} Document: 1", target) for text, target in zip(examples['text'], examples['doc_id'])]
            else:
                inputs = [truncate_doc(f"{example} Document: 1") for example in examples['text']]
            for input in inputs:
                for key in input:
                    model_inputs[key].append(input[key])
        return model_inputs

    tokenized_ds= ds.map(preprocess_function, batched=True)
    return tokenized_ds

def store_model(out_path, model, tokenizer):

    makedirs(out_path, exist_ok=True)

    tokenizer.save_pretrained(out_path)
    try:
        model.save_pretrained(out_path) 
    except:
        for param in model.parameters(): 
            param.data = param.data.contiguous()
        model.save_pretrained(out_path)

def store_data(out_path, ds, val_qrels, test_qrels=None):

    makedirs(out_path, exist_ok=True)

    ds.save_to_disk(out_path)

    json.dump(val_qrels, open(f'{out_path}/valid_qrels.json', 'w'))
    if test_qrels != None:
        json.dump(test_qrels, open(f'{out_path}/test_qrels.json', 'w'))


def main():
    print('start script')
    args = parser.parse_args()
    if args.ratio == None:
        args.ratio = "None"
    print(args)

    print('Reading in data...')
    documents = json.load(open(args.documents))

    train_queries = json.load(open(args.train))
    val_queries = json.load(open(args.val))
    if args.test:
        test_queries = json.load(open(args.test))
    
    # How does the data look like?
    # 1. Indexing: Document -> Label = DocID
    # 2. Retrieval: Query -> Label = DocID

    print('Creating dataset...')
    if args.doc_ids == 'atomic':
        doc_id_fct = convert_doc_id_token
    elif args.doc_ids == 'semantic':
        semantic_doc_id_mapping = json.load(open(args.semantic_doc_id_mapping))
        doc_id_fct = partial(convert_doc_id_semantic, semantic_doc_id_mapping)
    elif args.doc_ids == 'naive':
        doc_id_fct = lambda x: str(x)
    else:
        raise NotImplementedError(f'Currently, only atomic, naive and semantic doc_ids are implemented. But you requested: {args.doc_ids}')
    index_data, all_doc_ids = create_index_data(documents, doc_id_fct)

    retrieval_data, ret_doc_ids  = create_retrieval_data(train_queries, doc_id_fct)
    val_retrieval_data, val_ret_doc_ids  = create_retrieval_data(val_queries, doc_id_fct)
    complete_val_retrieval_data, val_qrels, _ = create_test_data(val_queries, doc_id_fct)
    print(len(all_doc_ids))
    all_doc_ids = all_doc_ids + ret_doc_ids + val_ret_doc_ids
    all_doc_ids = list(set(all_doc_ids))
    print(len(all_doc_ids))
    if test_queries:
        test_retrieval_data, test_qrels, test_doc_ids = create_test_data(test_queries, doc_id_fct)
        all_doc_ids = list(set(all_doc_ids + test_doc_ids))
    print(len(all_doc_ids))

    if args.tokenizer:
        tok_path = args.tokenizer
    else:
        tok_path = args.model_path
    del documents

    tokenizer = AutoTokenizer.from_pretrained(tok_path, token=args.token)

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, token=args.token)
        model_type = "encoder-decoder"
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, token=args.token)
        model_type = "decoder-only"

        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if args.doc_ids == 'atomic':
        tokenizer, model = resize_vocab(all_doc_ids, model, tokenizer)

    print('Storing model and tokenizer...')
    store_model(get_model_dir(args, model.name_or_path), model, tokenizer)
    del model

    ds = create_dataset(index_data, retrieval_data, val_retrieval_data, complete_val_retrieval_data, test_retrieval_data, ratio=args.ratio)
    #random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
    #ds.save_to_disk(random_string)
    #ds = load_dataset(random_string, streaming=True)

    print('Tokenizing dataset...')
    tokenized_ds = tokenize_data(ds, tokenizer, args, model_type)

    print('Storing dataset...')
    dataset_dir = get_dataset_dir(args)
    store_data(dataset_dir, tokenized_ds, val_qrels, test_qrels=test_qrels)
    json.dump(vars(args), open(f'{dataset_dir}/preprocessing_config.json', 'w'))
    
    print('Created Dataset!')
    
    

if __name__ == "__main__":
    main()
