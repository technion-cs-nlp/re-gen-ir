try:
    import comet_ml
    comet_installed = True
except ModuleNotFoundError:
    print("comet_ml is not installed.")
    comet_installed = False
from transformers import Seq2SeqTrainer, Trainer
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

import numpy as np
import torch


# https://stackoverflow.com/a/75381393
def topk(array, k, axis=-1, sorted=True):
    # Use np.argpartition is faster than np.argsort, but do not return the values in order
    # We use array.take because you can specify the axis
    partitioned_ind = (
        np.argpartition(array, -k, axis=axis)
        .take(indices=range(-k, 0), axis=axis)
    )
    # We use the newly selected indices to find the score of the top-k values
    partitioned_scores = np.take_along_axis(array, partitioned_ind, axis=axis)
    
    if sorted:
        # Since our top-k indices are not correctly ordered, we can sort them with argsort
        # only if sorted=True (otherwise we keep it in an arbitrary order)
        sorted_trunc_ind = np.flip(
            np.argsort(partitioned_scores, axis=axis), axis=axis
        )
        
        # We again use np.take_along_axis as we have an array of indices that we use to
        # decide which values to select
        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
    else:
        ind = partitioned_ind
        scores = partitioned_scores
    
    return scores, ind


def remove_tokens(label_list, forbidden_tokens):
    return list(filter(lambda x: x not in forbidden_tokens, label_list))

def postprocess(text):
    return [[t.strip()] for t in text]


def train(args, ds, model, tokenizer, training_args, data_collator):

    def compute_metrics(eval_preds):
        if args.use_comet and comet_installed:
            experiment = comet_ml.get_global_experiment()
        else:
            experiment = None
            
        preds, labels = eval_preds
        # shape N x label_length x |V|
        if isinstance(preds, tuple):
            preds = preds[0]

        if args.doc_ids == 'atomic':
            hits_at_1 = 0
            hits_at_10 = 0

            if args.model_type == 'encoder-decoder':
                _, top_k_indices = topk(preds[:,0,:], 10, 1)                

            else:
                _, top_k_indices = topk(preds, 10, 1)
            for example, label in zip(top_k_indices, labels):
                if type(label) in [list, np.ndarray]:
                    filtered_label = remove_tokens(label, tokenizer.all_special_ids+[-100])
                    if len(filtered_label) == 1:
                        filtered_label = filtered_label[0]
                else:
                    filtered_label = label

                if filtered_label == example[0]:
                    hits_at_1 += 1
                if filtered_label in example:
                    hits_at_10 += 1
            
            result = {
                'hits@1': hits_at_1 / labels.shape[0],
                'hits@10': hits_at_10 / labels.shape[0]
            }

            if experiment:
                with experiment.context_manager("intermediate_valid"):
                    for example, label in zip(top_k_indices, labels):
                        text = tokenizer.decode(example)
                        metadata = {'label':tokenizer.decode(label)}
                        experiment.log_text(text=text,  metadata=metadata)

        else:
            hits_at_1 = 0
            result = {}
            if args.model_type == 'encoder-decoder':
                decoded_preds = postprocess(tokenizer.batch_decode(preds, skip_special_tokens=True))
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = postprocess(tokenizer.batch_decode(labels, skip_special_tokens=True))
                level_results = defaultdict(int)

                for pred, label in zip(decoded_preds, decoded_labels):
                    if pred == label:
                        hits_at_1 += 1
                    if args.doc_ids == 'semantic':
                        levels_pred = pred[0].split(' ')
                        levels_label = label[0].split(' ')
                        print(levels_pred, levels_label)
                        for i in range(min(len(levels_pred), len(levels_label))):
                            if levels_pred[i] == levels_label[i]:
                                level_results[i] += 1
                            else:
                                break

                result['hits@1'] = hits_at_1 / labels.shape[0]
                if args.doc_ids == 'semantic':
                    for k, v in level_results.items():
                        result[f'level_{k}_hits'] = v / labels.shape[0]
                
                if experiment:
                    with experiment.context_manager("intermediate_valid"):
                        for text, label in zip(decoded_preds, decoded_labels):
                            metadata = {'label': label}
                            experiment.log_text(text=text,  metadata=metadata)

            else:
                raise NotImplementedError('Only encoder-decoder models are supported for semantic and naive doc ids.')


        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    if args.model_type == 'encoder-decoder':
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["valid"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["valid"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    if training_args.resume_from_checkpoint:
        trainer.train(training_args.resume_from_checkpoint)
    else:
        trainer.train()
