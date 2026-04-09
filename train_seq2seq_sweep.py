import argparse

import torch
import torch.nn as nn

from forecast_sweep_common import add_common_args, run_sweep


class Seq2SeqForecaster(nn.Module):
    def __init__(self, pred_len, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder_cell = nn.GRUCell(input_size=1, hidden_size=hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        seq = x[:, 0, :].unsqueeze(-1)
        _, h_n = self.encoder(seq)
        h_t = h_n[-1]

        decoder_input = torch.zeros(seq.size(0), 1, device=x.device)
        outputs = []
        for _ in range(self.pred_len):
            h_t = self.decoder_cell(decoder_input, h_t)
            step = self.out(h_t)
            outputs.append(step)
            decoder_input = step

        return torch.cat(outputs, dim=1)


def make_model(input_len, pred_len, args, device):
    _ = input_len
    return Seq2SeqForecaster(
        pred_len=pred_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)


def make_model_config(args, input_len, pred_len):
    return {
        "model_type": "seq2seq_gru_decoder",
        "input_len": input_len,
        "pred_len": pred_len,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Seq2Seq GRU-decoder sweep for delta forecasting")
    add_common_args(
        parser,
        default_output_dir="outputs_seq2seq_sweep",
        default_checkpoint_name="seq2seq_delta_huber_best.pth",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args, model_factory=make_model, model_config_factory=make_model_config)
