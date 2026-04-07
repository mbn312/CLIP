import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from datasets import load_dataset


class FashionMNIST(Dataset):
    def __init__(self, train=True):
        self.dataset = load_dataset("fashion_mnist", trust_remote_code=False)

        self.transform = T.ToTensor()

        if train:
            self.split = "train"
        else:
            self.split = "test"


        self.captions = {0: "An image of a t-shirt/top",
                        1: "An image of trousers",
                        2: "An image of a pullover",
                        3: "An image of a dress",
                        4: "An image of a coat",
                        5: "An image of a sandal",
                        6: "An image of a shirt",
                        7: "An image of a sneaker",
                        8: "An image of a bag",
                        9: "An image of an ankle boot"}


    def __len__(self):
        return self.dataset.num_rows[self.split]

    def __getitem__(self,i):
        img = self.dataset[self.split][i]["image"]
        img = self.transform(img)

        cap, mask = tokenizer(self.captions[self.dataset[self.split][i]["label"]])

        mask = mask.unsqueeze(0).repeat(mask.shape[0], 1)

        return {"image": img, "caption": cap, "mask": mask}
    

def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    if encode:
        encoded = list((chr(2) + text + chr(3)).encode("utf-8"))  # Adding SOT and EOT tokens
        if len(encoded) > max_seq_length:
            raise ValueError(
                f"Text is too long for max_seq_length={max_seq_length}: "
                f"got {len(encoded)} encoded bytes including special tokens."
            )

        out = torch.tensor(
            encoded + [0] * (max_seq_length - len(encoded)),
            dtype=torch.int32,
        )
        mask = torch.zeros(max_seq_length, dtype=torch.int32)
        mask[:len(encoded)] = 1
    else:
        if mask is None:
            raise ValueError("mask is required when encode=False.")

        if mask.ndim == 2:
            mask = mask[0]
        elif mask.ndim != 1:
            raise ValueError("mask must be 1D or 2D when decode=False.")

        valid_length = int(mask.ne(0).sum().item())
        out = [chr(int(x)) for x in text[1:valid_length-1]]
        out = "".join(out)
        mask = None

    return out, mask
