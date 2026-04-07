# Technical Overview

This document explains how the repository works at the source-code level and how it relates to the linked notebook and tutorial article.

## Project Context

The public repository is a small CLIP-style learning project centered on FashionMNIST. The code in [`training.py`](../training.py), [`model/model.py`](../model/model.py), and [`data/data_utils.py`](../data/data_utils.py) is the primary source of truth for the runnable project.

Two external resources provide additional context:

- The [Colab notebook](https://colab.research.google.com/drive/1E4sEg7RM8HBv4PkIhjWZuwCXXbB_MinS?usp=sharing) mirrors the implementation in notebook form and includes saved outputs from a CPU run.
- The [Medium tutorial](https://medium.com/correll-lab/building-clip-from-scratch-68f6e42d35f4), published on May 16, 2024, walks through the model architecture and uses the same FashionMNIST example.

The three sources are closely aligned, with a few small differences worth documenting:

- The notebook saves checkpoints to `../clip.pt`, while [`training.py`](../training.py) saves to `./clip.pt`.
- The notebook includes a zero-shot ranking example; the Python entry point only trains and runs the template-based test pass.
- The notebook records `Model Accuracy: 84 %`, while the Medium article summarizes the result as "around 85%".

## End-To-End Data Flow

The project trains on image-text pairs created from FashionMNIST labels.

1. [`FashionMNIST`](../data/data_utils.py) loads the Hugging Face `fashion_mnist` dataset with `trust_remote_code=False`.
2. Each numeric class label is converted into a fixed caption string such as `"An image of a sneaker"`.
3. The `tokenizer()` function encodes the caption into integer byte values, prepends a start-of-text token with value `2`, appends an end-of-text token with value `3`, and pads to `max_seq_length = 32` with zero bytes.
4. The tokenizer also builds a 1D mask marking non-padding tokens and raises a clear `ValueError` if the encoded text would exceed `max_seq_length`.
5. The dataset expands that 1D mask to a square `32 x 32` attention mask for training samples, while the text encoder can also consume the original 1D token mask directly.
6. The training loop feeds image tensors, tokenized captions, and masks into the CLIP model.
7. The model produces normalized image and text embeddings, forms a batchwise similarity matrix, and optimizes a symmetric contrastive loss.

Each training sample returned by the dataset has:

- `image`: tensor with shape `[1, 28, 28]`
- `caption`: tensor with shape `[32]`
- `mask`: tensor with shape `[32, 32]`

## Model Components

### Positional Embedding

[`PositionalEmbedding`](../model/model.py) creates fixed sinusoidal positional encodings and stores them as a non-trainable buffer with `register_buffer`. The same class is reused by both the image and text encoders.

### Attention And Transformer Blocks

The transformer implementation is intentionally minimal:

- `AttentionHead` computes scaled dot-product self-attention
- `MultiHeadAttention` concatenates several attention heads and applies an output projection
- `TransformerEncoder` applies pre-layer normalization, multi-head attention, an MLP with GELU, and residual connections

The text path uses masks to suppress padded positions. It accepts either token masks shaped `[batch, seq]`, expanded attention masks shaped `[batch, seq, seq]`, or no mask at all, in which case padding is inferred from zero-valued tokens. The image path does not use a mask.

### Text Encoder

[`TextEncoder`](../model/model.py) contains:

- An embedding table of size `vocab_size x text_width`
- Fixed positional embeddings
- A stack of transformer encoder blocks
- A learned projection from text width into the shared embedding dimension

After the transformer stack, the encoder selects the end-of-text position by using the normalized token mask to find the last non-padding token. That token is projected into the multimodal embedding space and L2-normalized.

Important implication: this is not the original CLIP tokenizer or text tower. The vocabulary is only 256 symbols because the text path operates directly on UTF-8 byte values.

### Image Encoder

[`ImageEncoder`](../model/model.py) is a ViT-style encoder:

- Images are patchified with a convolution whose kernel size and stride both equal the patch size
- A learned class token is prepended to the patch sequence
- Fixed positional embeddings are added
- A stack of transformer encoder blocks processes the sequence
- The class token output is projected into the shared embedding dimension and normalized

With the default configuration, a `28 x 28` FashionMNIST image and `14 x 14` patches produce 4 image patches. After adding the class token, the image transformer sequence length is 5.

### CLIP Loss

[`CLIP`](../model/model.py) wraps the image and text encoders and optimizes a symmetric contrastive objective.

The forward pass:

1. Encodes the batch of images and texts
2. Computes cosine-similarity-style logits with a learned temperature scale
3. Creates labels `0..N-1` so each image is matched to the caption at the same batch index
4. Applies cross-entropy once over image-to-text logits and once over text-to-image logits
5. Returns the average of the two losses

This is the core CLIP idea implemented in a compact form.

## Training And Testing Behavior

[`training.py`](../training.py) defines two functions, `train()` and `test()`, then runs both when the file is executed as a script.

### Training

The training path:

- Instantiates the training split of `FashionMNIST`
- Wraps it in a `DataLoader`
- Creates the `CLIP` model and Adam optimizer
- Trains for 10 epochs using the defaults defined at the bottom of the file
- Saves `clip.pt` whenever the final batch loss of the current epoch improves on the previous best loss

Two details matter for anyone reading results:

- The save criterion is training loss, not validation loss.
- The logged loss is the last batch loss of each epoch, not an epoch average.

### Testing

The test path:

- Loads the test split
- Rebuilds the same model architecture and loads the saved checkpoint
- Embeds all 10 training caption templates once
- Compares each test image embedding against those 10 text embeddings
- Counts a prediction as correct when the highest-similarity caption matches the expected caption tensor

Because the candidate texts are the same label templates used during training, this is best understood as closed-set evaluation on FashionMNIST classes.

## Notebook Behavior

[`notebooks/CLIPModel.ipynb`](../notebooks/CLIPModel.ipynb) follows the same structure as the code modules:

- imports
- positional embedding
- attention and transformer blocks
- tokenizer
- text encoder
- image encoder
- CLIP model
- dataset
- training parameters
- loading dataset
- training
- testing
- zero-shot classification

The saved outputs show:

- example tensor shapes from the dataset
- a CPU training run for 10 epochs
- `Model Accuracy: 84 %`
- a top-5 ranking example in the zero-shot section

The zero-shot section is still constrained to the FashionMNIST label set. It switches from prompt templates to raw class names such as `"dress"` and `"bag"`, but it does not evaluate on unseen categories.

## Default Hyperparameters

The default configuration in [`training.py`](../training.py) is:

| Parameter | Value |
| --- | --- |
| `emb_dim` | `32` |
| `vit_width` | `9` |
| `img_size` | `(28, 28)` |
| `patch_size` | `(14, 14)` |
| `n_channels` | `1` |
| `vit_layers` | `3` |
| `vit_heads` | `3` |
| `vocab_size` | `256` |
| `text_width` | `32` |
| `max_seq_length` | `32` |
| `text_heads` | `8` |
| `text_layers` | `4` |
| `lr` | `1e-3` |
| `epochs` | `10` |
| `batch_size` | `128` |

These values keep the model small enough to run as a tutorial example.

## Differences From Canonical CLIP

This project is CLIP-like in objective and encoder structure, but it differs materially from the original CLIP work:

- Training data is FashionMNIST rather than a large image-text corpus
- Text is tokenized as raw bytes rather than with BPE
- Both encoders are very small
- Evaluation is limited to 10 known FashionMNIST classes
- There is no pretraining checkpoint distribution, packaging, or large-scale benchmarking

Those differences are expected and intentional. The value of the repo is readability and educational clarity.

## Practical Caveats

- [`requirements.txt`](../requirements.txt) uses plain PyTorch version pins, but some platforms may still require a custom PyTorch install flow before the remaining dependencies are installed.
- The dataset is downloaded at runtime, so offline execution will fail unless the data is already cached.
- The repo has no automated tests and no formal release process at the time of writing.
