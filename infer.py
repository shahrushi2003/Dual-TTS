import argparse
import torch
import torchaudio

from models import TacotronMultiStream
from dataset import TextProcessor
from vocoder import get_vocoder


def synthesize(text, model, vocoder, device, max_steps=1000):
    model.eval()
    with torch.no_grad():
        seq = TextProcessor()(text).unsqueeze(0).to(device)
        mel, post, _, _ = model(seq, max_decoder_steps=max_steps)
        audio = vocoder(post.transpose(1,2))
    return audio[0].cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--model-path", type=str, default="dual_tts_model.pt")
    p.add_argument("--output", type=str, default="output.wav")
    p.add_argument("--max-steps", type=int, default=1000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacotronMultiStream().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    vocoder, device = get_vocoder(device)

    audio = synthesize(args.text, model, vocoder, device, max_steps=args.max_steps)
    torchaudio.save(args.output, audio, 22050)
    print(f"Saved audio to {args.output}")

if __name__ == "__main__":
    main()