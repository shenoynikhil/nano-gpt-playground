## Nano-GPT Implementation

Just playing around with Andrej Karpathy's [Nano-GPT implementation](https://github.com/karpathy/nanoGPT).

#### Environment
To setup the environment, you will need conda,
```bash
# create a conda env
conda create -n nano-gpt python=3.8
conda activate nano-gpt
pip install -r requirements.txt
```

#### Running the code
To run the code, you can use the following command,

```bash
python model.py
```

#### Core Ideas:
- The use of `torch.tril` and matrix multiplication to ensure autoregressive property for the decoder block. This property is not needed for the encoder block of a transformer.
- TODO: Add more details here.
