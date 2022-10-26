import numpy as np
import torch

class BinaryClassificationMetric:
    def __init__(self):
        self.tp = None
        self.tn = None
        self.fp = None
        self.fn = None

    def calc_confusion_matrix(self, pred, label):
        pred, label = pred.type(torch.bool), label.type(torch.bool)
        self.tp = sum(pred & label)
        self.tn = sum(~pred & ~label)
        self.fp = sum(pred & ~label)
        self.fn = sum(~pred & label)

    def get_accuracy(self):
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        return acc

    def get_recall(self):
        recall = self.tp / (self.tp + self.fn)
        return recall

    def get_precision(self):
        precision = self.tp / (self.tp + self.fp)
        return precision

    def get_f1_score(self):
        r, p = self.get_recall(), self.get_precision()
        f1 = 2 * r * p / (r + p)
        return f1


class MultiClassificationMetric:
    def __init__(self):
        pass

    def get_top_k(self, pred, label, k):
        
        rate_1 = 0
        rate_k = 0

        for p, l in zip(pred, label):
            l = l.cpu().numpy().astype(int)
            s = {c:prob.cpu().detach().numpy() for c, prob in enumerate(p)}
            s = sorted(s.items(), key=lambda x:x[1], reverse=True)
            s = [c for c, _ in s]
            
            if l==s[0]:
                rate_1 += 1
            
            if l in s[:k]:
                rate_k += 1

        top_1 = rate_1 / len(pred)
        top_k = rate_k / len(pred)

        return top_1, top_k