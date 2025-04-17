import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_dataset
import librosa

# TextProcessor maps characters to integer IDs
define_chars = "abcdefghijklmnopqrstuvwxyz '.,?!"
class TextProcessor:
    def __init__(self):
        self.char_to_id = {c: i for i, c in enumerate(define_chars)}

    def __call__(self, text: str) -> torch.LongTensor:
        text = text.lower().strip()
        seq = [self.char_to_id[c] for c in text if c in self.char_to_id]
        return torch.LongTensor(seq)

class LJSpeechDataset(Dataset):
    def __init__(self):
        self.dataset = load_dataset("lj_speech", split="train")
        self.text_processor = TextProcessor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        waveform, sr = torchaudio.load(item["file"])
        waveform = torchaudio.functional.resample(waveform, sr, 22050)
        text = self.text_processor(item["normalized_text"])
        mel = librosa.feature.melspectrogram(
            y=waveform.numpy()[0], sr=22050, n_fft=1024, hop_length=256, n_mels=80
        )
        return text, torch.FloatTensor(mel.T)

# Collate for batching

def collate_fn(batch): 
    texts, mels = zip(*batch)
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    mels_padded = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True)
    text_lens = [len(t) for t in texts]
    mel_lens = [m.size(0) for m in mels]
    return texts_padded, mels_padded, text_lens, mel_lens

