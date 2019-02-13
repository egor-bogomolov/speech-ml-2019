import torch
from torch import nn
import torch.nn.functional as F


class MultiInputRNN(nn.Module):

    def __init__(self, mfcc_features, fbank_features, hidden_size=128):
        super(MultiInputRNN, self).__init__()
        self.mfcc_features = mfcc_features
        self.fbank_features = fbank_features
        self.hidden_size = hidden_size
        self.lstm_mfcc = nn.LSTM(mfcc_features, hidden_size)
        self.lstm_fbank = nn.LSTM(fbank_features, hidden_size)
        self.predict_mfcc = nn.Linear(hidden_size, 2)
        self.predict_total = nn.Linear(2 * hidden_size, 2)

    def _get_hidden_state(self, batch_size):
        return torch.randn(1, batch_size, self.hidden_size), torch.randn(1, batch_size, self.hidden_size)

    def forward(self, sequence):
        batch_size = 1
        if len(sequence.size()) == 3:
            batch_size = sequence.size()[1]
        hidden_mfcc = self._get_hidden_state(batch_size)
        hidden_fbank = self._get_hidden_state(batch_size)
        sequence_mfcc, sequence_fbank = torch.split(sequence, [self.mfcc_features, self.fbank_features], -1)
        out_mfcc, _ = self.lstm_mfcc(
            sequence_mfcc.contiguous().view(-1, batch_size, self.mfcc_features),
            hidden_mfcc)
        out_fbank, _ = self.lstm_fbank(
            sequence_fbank.contiguous().view(-1, batch_size, self.fbank_features),
            hidden_fbank)
        prediction_mfcc = F.log_softmax(self.predict_mfcc(out_mfcc), dim=-1)
        prediction_total = F.log_softmax(self.predict_total(torch.cat((out_mfcc, out_fbank), dim=2)), dim=-1)

        return prediction_mfcc, prediction_total
