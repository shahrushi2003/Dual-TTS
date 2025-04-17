import torch
import torchaudio

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
vocoder = bundle.get_vocoder()

# Returns vocoder on correct device
def get_vocoder(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder.to(device)
    return vocoder, device