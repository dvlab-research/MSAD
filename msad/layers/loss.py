'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss 

import torch
import torch.nn as nn
import math

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=0).mean()
        return lossvalue

class L2_NORM(nn.Module):
    def __init__(self):
        super(L2_NORM, self).__init__()
    def forward(self, output, target):
        output_2 = output.pow(2).sum(dim=(2,3), keepdim=True).sqrt()
        target_2 = target.pow(2).sum(dim=(2,3), keepdim=True).sqrt()
        output_norm = output/output_2
        target_norm = target/target_2
        lossvalue = torch.norm(output_norm.view(-1)-target_norm.view(-1),p=2,dim=0)
        return lossvalue

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = L2()

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        return lossvalue

class L2Loss_NORM(nn.Module):
    def __init__(self):
        super(L2Loss_NORM, self).__init__()
        self.loss = L2_NORM()

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        return lossvalue


class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0

        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i]*EPE(output_, target_)
                lossvalue += self.loss_weights[i]*self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return  [lossvalue, epevalue]