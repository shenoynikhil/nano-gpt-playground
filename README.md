## Nano-GPT Implementation

This is based off Andrej Karpathy's [Nano-GPT implementation](https://github.com/karpathy/nanoGPT?tab=readme-ov-file).


### Setup

#### Environment
To setup the environment, you will need conda,
```bash
# create a conda env
conda create -n nano-gpt python=3.8
conda activate nano-gpt
pip install -r requirements.txt
```

#### Data
Running the python file will download the preprocess the data for the character-level language model training.

```bash
python data/prepare_data.py
```
