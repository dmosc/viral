# Viral: a multi-modal ML architecture to predict TikTok virality.

## Sources and exported checkpoints.

The following are relevant assets produced by this ML project:

* Training dataset: https://huggingface.co/datasets/rodmosc/viral
* Model trained with weighted BCE loss: https://huggingface.co/rodmosc/viral-weighted-loss
* Model trained with focal loss: https://huggingface.co/rodmosc/viral-focal-loss

## Setup

Follow this steps to run the project in your local machine:

### Environment and dependencies

Run the following commands at the root of the repository:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the following command to kick-off a training run:

```bash
python -m src.train_model
```

Run the following command to run the TikTok scraper used to produce the dataset:

```bash
python -m src.scripts.scrape_tiktok_data [--skip_n_examples]
```

Run the following command to export the dataset from raw data produced by the
scraper to Huggingface:

```bash
python -m src.scripts.compose_dataset
```

Project details can be found
[here](https://github.com/dmosc/viral/blob/main/docs/main.pdf).