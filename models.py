import torch
import torch.nn as nn

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, text: torch.LongTensor) -> torch.Tensor:
        embedded = self.embedding(text)
        outputs, _ = self.lstm(embedded)
        return self.fc(outputs)

# --- Multi-Stream Decoder ---
class MultiStreamDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        mel_dim: int = 80,
        max_len: int = 1000,
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.pos_embedding = nn.Parameter(torch.randn(max_len, hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.semantic_head = nn.Linear(hidden_size, mel_dim)
        self.acoustic_head = nn.Linear(hidden_size, mel_dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor):
        T, batch, _ = tgt.size()
        pos = self.pos_embedding[:T].unsqueeze(1).expand(-1, batch, -1)
        tgt = tgt + pos
        output = self.transformer_decoder(tgt, memory)
        return self.semantic_head(output), self.acoustic_head(output)

# --- Tacotron with Multi-Stream ---
class TacotronMultiStream(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size, mel_dim = 256, 80
        self.encoder = Encoder(embed_dim=hidden_size, hidden_size=hidden_size)
        self.mel_embedding = nn.Linear(mel_dim, hidden_size)
        self.decoder = MultiStreamDecoder(
            hidden_size=hidden_size,
            n_layers=4,
            n_heads=4,
            dim_feedforward=512,
            mel_dim=mel_dim,
        )
        self.postnet = nn.Sequential(
            nn.Conv1d(mel_dim, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Conv1d(512, mel_dim, kernel_size=5, padding=2),
        )

    def forward(self, text, mel=None, max_decoder_steps: int = 1000):
        # Encode text
        enc = self.encoder(text).transpose(0, 1)  
        if mel is not None:
            # Training with teacher forcing
            bs, T, _ = mel.size()
            device = mel.device
            start = torch.zeros(1, bs, self.decoder.mel_dim, device=device)
            mel_shifted = torch.cat([start, mel[:, :-1].transpose(0, 1)], dim=0)
            dec_in = self.mel_embedding(mel_shifted)
            sem, ac = self.decoder(dec_in, enc)
            sem, ac = sem.transpose(0,1), ac.transpose(0,1)
            ac_delayed = torch.cat([
                torch.zeros(bs,1,self.decoder.mel_dim, device=device), ac[:,:-1]
            ], dim=1)
            mel_out = sem + ac_delayed
            post = self.postnet(mel_out.transpose(1,2)).transpose(1,2)
            return mel_out, post, sem, ac
        # Inference mode follows similarly...  
        # (Omitted here for brevity, see notebook for full loop.)