import torch
import torch.nn as nn
import torch.nn.functional as F


class _Classifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight):
        self.weight.data = weight.clone()
    

class LinearClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        nn.init.kaiming_normal_(self.weight.data)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class CosineClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight) * self.scale


class L2NormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
    
    def forward(self, x):
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)


class LayerNormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-12, dtype=dtype)

    def forward(self, x):
        x = self.ln(x)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)
    


class _MahaClassifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)
        self.weight2 = nn.Parameter(torch.empty(feat_dim, feat_dim, dtype=dtype))
        # self.weight2.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)
        # self.bias = nn.Parameter(torch.full((num_classes,), 10, dtype=dtype))
        
    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight_input):
        mean, prec = weight_input[0], weight_input[1]
        self.weight.data = mean.clone()
        self.weight2.data = prec.clone()
    

class MahaClassifier(_MahaClassifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=0.1, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = nn.Parameter(torch.full((1,), scale, dtype=dtype))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        mean = F.normalize(self.weight, dim=-1)
        prec = F.normalize(self.weight2.T, dim=-1).T
        #bias = self.bias

        expanded_x = x.unsqueeze(1).expand(x.size(0), mean.size(0), x.size(1))
        delta = expanded_x - mean
        product = torch.matmul(delta, prec) #* delta
        output = product.sum(dim=-1)
        logit = -output #* self.scale #+ bias
        #print(logit[0])
        return logit

        