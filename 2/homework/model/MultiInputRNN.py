import torch
from torch import nn
import torch.nn.functional as F


class MultiInputRNN(nn.Module):

    def __init__(self, mfcc_features, fbank_features, hidden_size=128):
        super(MultiInputRNN, self).__init__()
        self.lstm_mfcc = nn.LSTM(mfcc_features, hidden_size)
        self.lstm_fbank = nn.LSTM(fbank_features, hidden_size)
        self.predict_mfcc = nn.Linear(hidden_size, 1)
        self.predict_total = nn.Linear(2 * hidden_size, 1)

    def forward(self, mfcc, fbank):
        _, hidden_mfcc = self.lstm_mfcc(mfcc)
        _, hidden_fbank = self.lstm_fbank(fbank)

        prediction_mfcc = F.softmax(self.predict_mfcc(hidden_mfcc))
        prediction_total = F.softmax(self.predict_total(torch.cat((hidden_mfcc, hidden_fbank), dim=1)))

        return prediction_mfcc, prediction_total
