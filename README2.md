# Disclaimer: Forked Repository for VALSE Paper Extension

This repository is a **fork** of the original codebase used to produce the results presented in the VALSE (Vector-based Assessment of Linquistic Systematicity and Entailment) paper.

We have utilized this forked version to test an **additional linguistic phenomenon** not covered in the original VALSE study. Our modifications and new experiments are contained within this fork.

## Locating Our Results

To find the specific results related to our investigation of the additional linguistic phenomenon, please navigate to the results-folder. There you will find:

* Pairwise Accuracy ranking, accuracy, precision, foil precision and AUROC*100 from Lxmert 
* Perplexity scores from GPT1
* Perplexity scores from GPT2

## Reproducing Our Results

To reproduce the results from our experiments on the additional linguistic phenomenon, please follow these steps:

1.  **Prerequisites**: Set up the environment by following the instructions in "environment_minimal_yaml". This setup assumes windows, for other operative systems the commands might be slightly different.

2.  **Data Preparation**:
    * Download the additional dataset from [link](https://dreamdragon.github.io/PennAction/). Locate the frames-folder, and place it inside `data/temporal-stage/".
  
3.  **Running the Experiment**:
    * Run lxmert_valse_eval.py to run test on tthe VLM model.
    * Run Unimodal_valse_eval.py to run test on unimodal models. To specify which, configure the "which"-variable inside the file.

Please refer to the original `README.md` for instructions on reproducing the original VALSE paper results. Our additions are specifically focused on the new linguistic phenomenon investigated.

## Where our calculations come from
* If you should be curious of how we created the temporal_stage.json file, how we selected the images and annotated them or how we calculated the AUROC value, feel free to check out our "workshed"-repository: [link](https://github.com/robinsvahn/ActionState)