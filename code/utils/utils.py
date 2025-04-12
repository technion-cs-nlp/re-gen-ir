import numpy as np
import torch
import random

def get_model_dir(args, model_name=None):
    if model_name is None:
        data_dir = args.data
        model_name = args.model_name
    else:
        data_dir = args.output
    return f'{data_dir}/{args.doc_ids}/models/{model_name}' 

def get_dataset_dir(args):
    if 'output' in args:
        return f'{args.output}/{args.doc_ids}/{args.ratio}'
    elif 'data' in args:
        return f'{args.data}/{args.doc_ids}/{args.ratio}'
    
def set_random_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 10000)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def convert_doc_id_token(doc_id):
    if type(doc_id) not in [str, int]:
        raise TypeError('Only str and int are supported as doc ids.')
    doc_id = str(doc_id).strip()
    return f'@DOC_ID_{doc_id}@'
