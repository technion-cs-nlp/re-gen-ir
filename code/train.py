try:
    import comet_ml
except ModuleNotFoundError:
    pass
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, HfArgumentParser, Seq2SeqTrainingArguments, AutoModelForSequenceClassification, DataCollatorForSeq2Seq
from training.trainer import train
from utils.utils import *
from torch.nn.utils.rnn import pad_sequence

import json

from eval import evaluate

def add_arguments(parser):
    parser.add_argument('-d', '--data', help='path of the pre-processed data (including train, eval/dev)', required=True)
    parser.add_argument('-t', '--model-type', help='are we training a encoder-decoder or only a decoder type model', 
                        required=False, default='encoder-decoder', choices=['encoder-decoder', 'decoder'])
    parser.add_argument('-m', '--model-name', help='model identifier that was used during pre-processing', required=True)
    parser.add_argument('-c', '--use-comet', help='use coment_ml to track experiments or not', required=False, default=False, action='store_true')
    parser.add_argument('-r', '--ratio', help='the ratio of indexing to retrieval examples', default=1)
    parser.add_argument('--doc-ids', help='the method of creating doc ids', default='atomic', choices=['naive', 'semantic', 'atomic'])
    parser.add_argument('-o', '--output-dir', help='the output directory of the model', default='experiment')
    parser.add_argument('--test-batch-size', help='batch size used for evaluation', default=2)
    parser.add_argument('-s', "--everything-seed", help='sets the seed to a predetermined value', type=int)
    parser.add_argument('--remove', help='a list of docs to remove from the training, syntax: str "D1, D2, ..."', type=str, default='')

parser_seq2seq = HfArgumentParser(Seq2SeqTrainingArguments, prog='Team Seq2Se2 Trainer', description='Train a GenIR model')
add_arguments(parser_seq2seq)
parser_trainer = HfArgumentParser(Seq2SeqTrainingArguments, prog='Team Default Trainer', description='Train a GenIR model')
add_arguments(parser_trainer)

def main():
    try:
        args = parser_trainer.parse_args()
    except:
        args = parser_seq2seq.parse_args()
    
    if args.model_type == 'encoder-decoder':
        print('Switching to Seq2Seq Trainer, because model_type="encoder-decoder".')
        args = parser_seq2seq.parse_args()
        parser = parser_seq2seq
    elif args.model_type == 'decoder':
        print('Using Default Trainer, because model_type="decoder"')
        args = parser_trainer.parse_args()
        parser = parser_trainer
    else:
        raise ValueError(f'Unknown model_type detected. {args.model_type} is not supported.')

    if args.everything_seed:
        seed = args.everything_seed
        print("Setting seed to", seed)
        set_random_seed(seed)
    else:
        seed = set_random_seed()
        print("Setting seed to", seed)
    print("Loading dataset...")
    data_dir = get_dataset_dir(args)
    ds = load_from_disk(data_dir)
    if args.remove != "":
        print(len(ds['train'])) 
        removal_documents = args.remove.split(',')
        removal_documents = [doc.strip() for doc in removal_documents]    
        print('removing from corpus: ',removal_documents)
        ds['train'] = ds['train'].filter(lambda x: x['doc_id'] not in removal_documents )
        print(len(ds['train'])) 
    
    min_valid = min(1024, len(ds['valid']))
    ds['valid'] = ds['valid'].select(range(min_valid))
    print("Loading model and tokenizer...")
    print(ds['train'].select([0]))
    print(ds['valid'].select([0]))
    #exit()

    model_dir = get_model_dir(args)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    training_args, _ = parser.parse_args_into_dataclasses(args=None, return_remaining_strings=False, look_for_args_file=False)
    if args.model_type == "encoder-decoder":
        if training_args.resume_from_checkpoint:
            model = AutoModelForSeq2SeqLM.from_pretrained(training_args.resume_from_checkpoint)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    elif args.model_type == "decoder":
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        del model.config.label2id
        del model.config.id2label

        def data_collator(features):
            input_ids = [torch.tensor(feature["input_ids"]) for feature in features]

            batch_size = len(input_ids)
            labels_tensor = torch.full((batch_size,), tokenizer.pad_token_id, dtype=torch.long)

            for i, feature in enumerate(features):
                if len(feature["labels"]) == 1:
                    labels_tensor[i] = torch.tensor(feature["labels"])
                else:
                    labels_tensor[i] = torch.tensor(feature["labels"][1])

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

            return {"input_ids": input_ids, "labels": labels_tensor}

    
    
    training_args.seed = seed
#    if training_args.resume_from_checkpoint:
#        resume = training_args.resume_from_checkpoint
#        training_args = Seq2SeqTrainingArguments(model_dir)
#        training_args.resume_from_checkpoint = resume

    train(args, ds, model, tokenizer, training_args, data_collator)

    if not args.load_best_model_at_end:
        print('Warning: load_best_model_at_end is set to False. The last model checkpoint will be evaluated on the test set.')
    else:
        print(f'Evaluating the best model according to:', args.metric_for_best_model)

    print('Evaluation on Validation Set:')
    qrels = json.load(open(f"{data_dir}/valid_qrels.json"))
    evaluate(args, ds['complete_valid'], qrels, model, tokenizer, args.output_dir, data_prefix='valid')

    if 'test' in ds:
        print('Evaluation on Test Set:')
        qrels = json.load(open(f"{data_dir}/test_qrels.json"))
        evaluate(args, ds['test'], qrels, model, tokenizer, args.output_dir, data_prefix='test')
    else:
        print('No Test Set found.')

if __name__ == '__main__':
    main()
