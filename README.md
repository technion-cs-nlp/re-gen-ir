# Reverse-Engineering the Retrieval Process in GenIR Models

This is the repository for our project "Reverse-Engineering the Retrieval Process in GenIR Models". The code for training, evaluation and analysis can be found in the directory `code`. The website for our paper is hosted [here](https://technion-cs-nlp.github.io/re-gen-ir/index.html).

## Models

The models for our publication are hosted on Huggingface (each model is also linked to the dataset used for training and evaluation):

| Model        | Huggingface URL                                                         |
| ------------ | ----------------------------------------------------------------------- |
| NQ10k        | [DSI-large-NQ10k](https://huggingface.co/AnReu/DSI-large-NQ10k)         |
| NQ100k       | [DSI-large-NQ100k](https://huggingface.co/AnReu/DSI-large-NQ100k)       |
| NQ320k       | [DSI-large-NQ320k](https://huggingface.co/AnReu/DSI-large-NQ320k)       |
| Trivia-QA    | [DSI-large-TriviaQA](https://huggingface.co/AnReu/DSI-large-TriviaQA)   |
| Trivia-QA QG | [DSI-large-TriviaQA QG](https://huggingface.co/AnReu/DSI-large-TriviaQA-QG) |

## Model Usage

[Here](https://colab.research.google.com/drive/114TVXj2eqp0CrSUtlaBa8WTnyAsN1nsF?usp=sharing) is a complete example of using the models for retrieval.

Quick example usage:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = 'AnReu/DSI-large-NQ10k'
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

query = 'this is a test query'
input_ids = tokenizer(query, return_tensors='pt').input_ids
decoder_input_ids = torch.zeros([1,1], dtype=torch.int64)
output = model(input_ids, decoder_input_ids=decoder_input_ids)
```

### Citation
```
@inproceedings{Reusch2025Reverse,
  author = {Reusch, Anja and Belinkov, Yonatan},
  title = {Reverse-Engineering the Retrieval Process in GenIR Models},
  year = {2025},
  isbn = {9798400715921},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3726302.3730076},
  doi = {10.1145/3726302.3730076},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages = {668–677},
  numpages = {10},
  location = {Padua, Italy},
  series = {SIGIR '25}
}
```
