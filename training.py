import torch
from data.data_utils import FashionMNIST, tokenizer
from torch.utils.data import DataLoader
from model.model import CLIP
from torch.optim import Adam
import numpy as np

DEVICE = torch.device("cpu")

def train(batch_size, emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers, lr, epochs, model_location):

    dataset = FashionMNIST()
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    model = CLIP(emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    
    best_loss = np.inf

    for epoch in range(epochs):
        for _, data in enumerate(dataloader, 0):
            img, cap, mask = data["image"].to(DEVICE), data["caption"].to(DEVICE), data["mask"].to(DEVICE)
            loss = model(img,cap,mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Batch Loss: {loss.item():.3f}")

        # Saves model if it performed better than the previous best
        if loss.item() <= best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), model_location)
            print("Model Saved.")


def test(batch_size, emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers, model_location):
    test_set = FashionMNIST(train=False)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    model = CLIP(emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers).to(DEVICE)
    model.load_state_dict(torch.load(model_location, map_location=DEVICE))

    # Getting dataset captions to compare images to
    text = torch.stack([tokenizer(x)[0] for x in test_set.captions.values()]).to(DEVICE)
    mask = torch.stack([tokenizer(x)[1] for x in test_set.captions.values()])
    mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(DEVICE)

    correct, total = 0,0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data["image"].to(DEVICE), data["caption"].to(DEVICE)
            image_features = model.image_encoder(images)
            text_features = model.text_encoder(text, mask=mask)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * (image_features @ text_features.T)).softmax(dim=-1)
            _, indices = torch.max(similarity,1)
            pred = torch.stack([tokenizer(test_set.captions[int(i)])[0] for i in indices]).to(DEVICE)
            correct += int(sum(torch.sum((pred==labels),dim=1)//len(pred[0])))
            total += len(labels)

    print(f'\nModel Accuracy: {100 * correct // total} %')    


if __name__=="__main__":
    emb_dim = 32
    vit_width = 9
    img_size = (28,28)
    patch_size = (14,14)
    n_channels = 1
    vit_layers = 3
    vit_heads = 3
    vocab_size = 256
    text_width = 32
    max_seq_length = 32
    text_heads = 8
    text_layers = 4
    lr = 1e-3
    epochs = 10
    batch_size = 128
    model_location = "./clip.pt"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", DEVICE, f"({torch.cuda.get_device_name(DEVICE)})" if torch.cuda.is_available() else "")

    train(batch_size, emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers, lr, epochs, model_location)

    test(batch_size, emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers, model_location)