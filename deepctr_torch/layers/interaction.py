import torch.nn as nn
import torch
class FM(nn.Module):
    def __init__(self):
        super(FM,self).__init__()
    def forward(self, inputs):
        fm_input = inputs
        #print(fm_input,1,fm_input.shape)
        #print(torch.sum(fm_input, dim=1, keepdim=True),2,torch.sum(fm_input, dim=1, keepdim=True).shape)
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        #print(cross_term,"-"*20)
        return cross_term#torch.FloatTensor(cross_term)