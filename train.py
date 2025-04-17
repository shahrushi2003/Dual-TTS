import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models import TacotronMultiStream
from dataset import LJSpeechDataset, collate_fn


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacotronMultiStream().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    dataset = LJSpeechDataset()
    train_size = int(len(dataset) * args.train_split)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        model.train()
        total_train = 0
        for texts, mels, _, _ in train_loader:
            texts, mels = texts.to(device), mels.to(device)
            optimizer.zero_grad()
            mel_out, post_out, _, _ = model(texts, mel=mels)
            loss = criterion(mel_out, mels) + criterion(post_out, mels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)

        model.eval()
        total_val = 0
        with torch.no_grad():
            for texts, mels, _, _ in val_loader:
                texts, mels = texts.to(device), mels.to(device)
                mel_out, post_out, _, _ = model(texts, mel=mels)
                total_val += (criterion(mel_out, mels) + criterion(post_out, mels)).item()
        avg_val = total_val / len(val_loader)

        print(f"Epoch {epoch}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), args.output)
            print(f"Saved best model: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--output", type=str, default="dual_tts_model.pt")
    args = parser.parse_args()
    train(args)