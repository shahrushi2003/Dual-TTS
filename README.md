# Dual-TTS

**Dual-TTS** is a Tacotron-like multi-stream text-to-speech (TTS) model that combines semantic and acoustic decoding streams to produce high-quality mel spectrograms. It supports autoregressive inference and is trained on the LJSpeech dataset.

---

## Features

- Transformer-based multi-stream decoder  
- Tacotron-style encoder  
- Postnet for mel refinement  
- LJSpeech integration via Huggingface `datasets`  
- WaveRNN vocoder from `torchaudio`  

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset

The model uses [LJSpeech](https://huggingface.co/datasets/lj_speech) via the `datasets` library.

---

## Training

```bash
python train.py
```

This will train the model for 1 epoch (modify `num_epochs` inside `train.py` to train longer).

---

## Inference

```bash
python infer.py
```

Generates speech from a given input string. Output will be saved to `output.wav`.

---

## Results

![image](https://github.com/user-attachments/assets/34a583bb-8403-4e3f-86f7-4d6f0f0b4320)

---

## Example

```python
from infer import synthesize

audio_path = synthesize("Hello world, this is synthesized speech!")
print(f"Saved synthesized audio to: {audio_path}")
```
