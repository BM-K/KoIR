import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss():

    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

    def base(self, config, logits):
        if self.args.model == 'colbert':
            B = logits.size(0)
            labels = torch.zeros(B, dtype=torch.long, device=self.args.device)
            return self.criterion(logits, labels), labels

        elif self.args.model == 'simir':
            cosine_similarity, labels = logits
            return self.criterion(cosine_similarity, labels), labels
