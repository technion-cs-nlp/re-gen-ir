## Preprocessing

```python preprocess.py -d documents.json -t train_queries.json -v validation_queries.json --test test_queries.json -o output_dir -m hf-model```

The script will automatically infer the tokenizer from the model name, but you can speficy a certain tokenizer by using `--tokenizer tokenizer_name`.

Further Options:
- `-r` the ratio of indexing to retrieval examples, int for ratio, None for no increased/decreased ratio, default: 1
- `--max-length` maximum length of the input to the model, default: 128
- `--doc-ids {naive,semantic,atomic}` the method of creating doc ids, default: atomic
- `--prefix` prefix prepended to the input string (e.g., a prompt), default: None
- `--token` access token for gated HF models and tokenizers
- `--semantic-doc-id-mapping` mapping from doc ids to semantic doc ids

The script will takes a corpus split in document corpus and queries, and creates the dataset for GenIR. It creates the samples for the indexing and the retrieval tasks, re-arranges the corpus according to the ratio, and tokenizes the data before it is stored on disk.

## Training
```python train.py -d data_path -m hf_model -o experiment_output_dir```

The train script passes all options to the HuggingFace Trainer. 

Further Options:
- `--model-type {encoder-decoder,decoder}` whether we are training a encoder-decoder or only a decoder type model (default: encoder-decoder)
- `--use-comet` whether to use coment_ml to track experiments or not (default: False)
- `-r` the ratio of indexing to retrieval examples (default: 1)
- `--doc-ids {naive,semantic,atomic}` the method of creating doc ids (default: atomic)
- `--test-batch-size` batch size used for evaluation (default: 2)
- `--everything-seed` sets the seed to a predetermined value (default: None)
- `--remove` a list of doc ids to remove from the training, syntax: str "D1, D2, ..." (default: [])

## Evaluation

```python eval.py -d data_path -e EXPERIMENT_DIR```

Further Options:
- `--model-type {encoder-decoder,decoder}` whether we are training a encoder-decoder or only a decoder type model (default: encoder-decoder)
- `--use-comet` whether to use coment_ml to track experiments or not (default: False)
- `-r` the ratio of indexing to retrieval examples (default: 1)
- `--doc-ids {naive,semantic,atomic}` the method of creating doc ids (default: atomic)
- `--experiment-dir` the output directory of the model training, i.e., where to load the model from
- `--test-batch-size` batch size used for evaluation (default: 2)

## Analyses

The following scripts require validation query-document pairs and a json containing the validation queries that whose relevant documents where correctly identified by the model. Very small examples of these files can be found in the subdirectory `sample_data`.

To reproduce the experiments from the paper: 

### Encoder Swapping

In order to run this experiment, training of one complete model with all documents and at least one other models trained on some documents less is required. 

```python analysis/swap_encoder.py --complete-model-path path --missing-docs-for-model missing_map.json --val-queries val_queries.json --correct-valid correct_valid.json -o output_path```

- `--complete-model-path` path to the complete model checkpoint, which was trained on the entire dataset
- `--missing-docs-for-model` path to the missing documents file. The file should contain a JSON dictionary, where the keys are model paths and the values are comma-separated strings of document IDs. The model path should point to the model which was trained on all documents except for the ones indicated in the comma seperated string list. 
- `--val-queries` path for the validation queries, json
- `--correct-valid` path to the correct_valid.json file, contains query doc pairs that the complete model ranked correctly (produced by the eval script)

### Length and Angle of residual stream, ranks of relevant documents, logit lens


```python analysis/eval_residual_rank.py --checkpoint model_checkpoint --val-queries sample_data/val_queries_sample.json --correct-valid sample_data/correct_valid.json -o output_path```

- `--checkpoint` checkpoint for the model to evaluate
- `--val-queries` path for the validation queries, json
- `--correct-valid` path to the correct_valid.json file, contains query doc pairs that the model ranked correctly (produced by the eval script)
- `-o` path to which the output files are written


### Zero/ Mean Patching of single stages

```python analysis/eval_patching_single_combis.py --checkpoint model_checkpoint --val-queries val_queries.json --correct-valid correct_valid.json -o output_path```

- `--checkpoint` checkpoint for the model to evaluate
- `--val-queries` path for the validation queries, json
- `--correct-valid` path to the correct_valid.json file, contains query doc pairs that the model ranked correctly (produced by the eval script)
- `-o` path to which the output files are written



### Zero/ Mean Patching total circuit

```python analysis/eval_patching_all_stages.py --checkpoint model_checkpoint --val-queries val_queries.json --test-queries test_queries.json   --qrels qrels_file -o output_path```

- `--checkpoint` checkpoint for the model to evaluate
- `--val-queries` path for the validation queries, json, used for computing mean activations
- `--test-queries` these are the queries for which we compute the results
- `--qrels` path for the test qrels
- `-o` path to which the output files are written

### Patching-in T5

```python analysis/eval_patching_t5.py --checkpoint model_checkpoint --val-queries val_queries.json  --qrels qrels_file -o output_path```

- `--checkpoint` checkpoint for the model to evaluate
- `--val-queries` path for the validation queries, json
- `--qrels` path for the validation qrels
- `--t5` which version of T5 to use, default: t5-large
- `-o` path to which the output files are written


### From where do neurons read

Depending on the number of data samples used for this analysis, this might take a while (~hours).

```python analysis/eval_neuron_reading.py --checkpoint model_checkpoint --val-queries val_queries.json --sim-type dot --correct-valid correct_valid.json -o output_path```


- `--checkpoint` checkpoint for the model to evaluate
- `--val-queries` path for the validation queries, json
- `--sim-type` similarity score to caluclate: cosine or dot
- `--correct-valid` path to the correct_valid.json file, contains query doc pairs that the model ranked correctly (produced by the eval script)
- `-o` path to which the output files are written

```python analysis/get_topk_neuron_readings.py --path output_path --sim-type dot```

- `--path` path of the all_queries_neuron_similarities files to evaluate 
- `--sim-type` similarity score to caluclate: cosine or dot
- `--topk` top k components to average or None for all, default: None


### From where does cross-attention read

```python analysis/eval_cr_attn_reading.py --checkpoint model_checkpoint --val-queries val_queries.json --sim-type dot --correct-valid correct_valid.json -o output_path```

- `--checkpoint` checkpoint for the model to evaluate
- `--val-queries` path for the validation queries, json
- `--sim-type` similarity score to caluclate: cosine or dot
- `--correct-valid` path to the correct_valid.json file, contains query doc pairs that the model ranked correctly (produced by the eval script)
- `-o` path to which the output files are written


### Cross-Attention Output

```python analysis/cross-attention-outputs.py --checkpoint model_checkpoint --val-queries val_queries.json --correct-valid correct_valid.json -o output_path```

- `--checkpoint` checkpoint for the model to evaluate
- `--val-queries` path for the validation queries, json
- `--correct-valid` path to the correct_valid.json file, contains query doc pairs that the model ranked correctly (produced by the eval script)
- `-o` path to which the output files are written


### Visualizations

Jupyter Notebooks for the plots in the paper can be found in the subdirectory `analysis/visualizations`.