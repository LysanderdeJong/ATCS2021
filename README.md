# ATCS2021
Code for the ATCS assignment 2: Learning sentence representations

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#training">Training</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
      </ul>
    </li>
    <li>
      <a href="#demo-and-analysis">Demo and Analysis</a>
    </li>
    <li>
      <a href="#results">Results</a>
      <ul>
        <li><a href="#validation">Validation</a></li>
        <li><a href="#test">Test</a></li>
      </ul>
    </li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

In this excersie we attempted to replicate some of the results from [Conneau et al. (2017)](https://arxiv.org/abs/1705.02364).

We created four models which create sentence embeddings from pre-trained GloVe word embeddings:
* Words: This model takes the mean of the word embeddings to represent the sentence embedding.
* LSTM: Uses a LSTM of size 2048, with the last output of the LSTM used as the sentence representation.
* BiLSTM: Uses a bidirectional LSTM, with the last output for both direction concatenated as the sentence embedding.
* BiLSTM_Max: Uses a bidirectional LSTM, but takes the max over each stage of the model as the sentence embedding.

### Built With
This project is based on the following deep learning frameworks:
* [Pytorch](https://pytorch.org/)
* [TorchText](https://pytorch.org/text/stable/index.html)
* [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/)


<!-- GETTING STARTED -->
## Getting Started

Clone this github repo using:
```sh
git clone https://github.com/LysanderdeJong/ATCS2021
cd ATCS2021
```

Install the provided envoriment.yml using [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) as follows:
```sh
conda env create -f environment.yml
```
Activate the envorment using:
```sh
conda activate atcs
```
You should now have a working enviroment.

Since we are using [Spacy](https://spacy.io/) as our tokenizer, install it using:
```sh
python -m spacy download en_core_web_sm
```
   
<!-- USAGE EXAMPLES -->
## Usage
This section will briefly detail how to train the models from scratch and how to evaluate them on a veriety of NLP tasks.

### Training
Training a model from scratch is as simple as excuting:
```sh
python train.py --model "model_name" --gpus 1
```
You can choose from the following model aliases: 'words', 'lsmt', 'bilsmt' and 'bilsmt_max'.
The models will automatically be trained on the gpu. Should you whish to observe the progress, use the optional '--progress-bar' argument.

You can always type:
```sh
python train.py --help
```
to see what other switches exist.

NOTE: If during the proces of downloading the dataset you get an extraction error, please axtract it manualy. At the moment torchtext has trouble automatically extracting the SNLI_1.0.zip file.

After training has finished you can find the model checkpoint in "./logs/lighting_logs", together with a tensorboard file. 
Tensorboard files to view the training progress can be downloaded [here](https://github.com/LysanderdeJong/ATCS2021/releases/download/v1.0/tensorboard_logs.7z).

### Evaluation
We evaluate our model using [SentEval](https://github.com/facebookresearch/SentEval). Visit their github and read their paper so get a better discription of each NLP taks.
In order to evaluate the model, we must first download the required evaluation framework:

1. Clone the repo and move to it directory
   ```sh
   git clone https://github.com/facebookresearch/SentEval
   cd SentEval/
   ```
2. Install [SentEval](https://github.com/facebookresearch/SentEval)
   ```sh
   python setup.py install
   ```
3. Download dataset for downstream task evaluation
   ```sh
   cd data/downstream/
   ./get_transfer_data.bash
   ```
   
The trained models can be avaluated using:
```sh
python eval.py --model model_name --checkpoint_path "path_to_checkpoint"
```
For the checkpoint you can use your own or download our from the releases section on github.

The program will fit an MLP to your sentence embeddings and uses it to evaluate various NLP tasks.
The accuracy on these taskes is stored in a dictionary which is saved to the current directory so it can quickly be retieved for later analysis.

This repo also includes these results in the "results_model.pt" files.

<!-- Demo and Analysis -->
## Demo and Analysis
We include a jupyter notebook demo in which we show our models on the NLI task. We also include more detailed results showing the different properties the setence embeddings possess.

The demo allows to to predict the entailment for any given premise and hypothesis. See [demo.ipynb](https://github.com/LysanderdeJong/ATCS2021/blob/main/demo.ipynb) for futher details.

Also included is a small [report](https://github.com/LysanderdeJong/ATCS2021/blob/main/report.pdf) which goes into greater detail on the performance of the various models.

<!-- RESULTS -->
## Results
Below are the results for the four models on the validation en test sets for the various tasks in [SentEval](https://github.com/facebookresearch/SentEval).

### Validation

| Model      | #params | SNLI |CR |  MR |  MPQA | SUBJ | SST2 | TREC | MRPC | SICK-E | SICK-R | STS14 | micor | macro |
| :---       |  :---:  |:---: |:---:|:---:|:---: |:---:| :---:|:---: |:---: | :---:  | :---:  | :---: | :---: | :---: |
| [Words](https://github.com/LysanderdeJong/ATCS2021/releases/download/v1.0/checkpoint_words.pth)      |    0    |65.87 |79.94|78.00|88.22 |91.75|79.01 |74.27 |72.79 |70.4    | 0.696  |0.55/0.56|82.54| 79.30 |
| [LSTM](https://github.com/LysanderdeJong/ATCS2021/releases/download/v1.0/checkpoint_lstm.pth)       |  19.3 M |72.78 |81.24|77.81|88.38 |86.04|70.76 |57.70 |71.66 |79.4    | 0.762  |0.49/0.47|79.27| 76.62 |
| [BiLSTM](https://github.com/LysanderdeJong/ATCS2021/releases/download/v1.0/checkpoint_bilstm.pth)    |  38.5 M |77.25 |84.01|80.41|89.19 |90.93|77.29 |82.01 |71.91 |84.8    | 0.840  |0.53/0.52|84.45| 82.57 |
| [BiLSTM-Max](https://github.com/LysanderdeJong/ATCS2021/releases/download/v1.0/checkpoint_bilstm_max.pth) |  38.5 M |84.39 |83.98|79.92|89.63 |92.83|81.54 |84.21 |74.93 |86.2    | 0.892  |0.69/0.67|85.47| 84.15 |

### Test
| Model      | #params | SNLI |CR |  MR |  MPQA | SUBJ | SST2 | TREC | MRPC | SICK-E | SICK-R | STS14 | micor | macro |
| :---       |  :---:  |:---: |:---:|:---:|:---: |:---:| :---:|:---: |:---: | :---:  | :---:  | :---: | :---: | :---: |
| [Words](https://github.com/LysanderdeJong/ATCS2021/releases/download/v1.0/checkpoint_words.pth)      |    0    |64.71 |78.49|77.47|87.86 |91.45|81.05 |81.6  |71.77 |75.22   | 0.763  |0.55/0.56|82.96| 80.61 |
| [LSTM](https://github.com/LysanderdeJong/ATCS2021/releases/download/v1.0/checkpoint_lstm.pth)       |  19.3 M |71.14 |79.52|77.12|87.93 |85.35|68.04 |47.4  |69.28 |80.70   | 0.793  |0.49/0.47|81.18| 74.42 |
| [BiLSTM](https://github.com/LysanderdeJong/ATCS2021/releases/download/v1.0/checkpoint_bilstm.pth)    |  38.5 M |75.86 |83.05|79.74|88.59 |90.44|79.02 |85.2  |71.71 |84.55   | 0.857  |0.53/0.52|84.84| 82.79 |
| [BiLSTM-Max](https://github.com/LysanderdeJong/ATCS2021/releases/download/v1.0/checkpoint_bilstm_max.pth) |  38.5 M |83.95 |82.73|79.03|88.91 |92.28|83.75 |89.8  |74.72 |85.95   | 0.888  |0.69/0.67|85.66| 84.65 |
