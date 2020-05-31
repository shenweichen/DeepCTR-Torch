import torch


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2, device='cpu'):
        """
        :param weight_list: list
        :param weight_decay: float.
        :param p: int.
        """
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.device = device

    def forward(self):
        weight_list = self.get_weight()
        reg_loss = self.regularization_loss(weight_list)
        return reg_loss

    def get_weight(self):
        weight_list = []
        for name, param in filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.model.named_parameters()):
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list):
        reg_loss = torch.zeros((1,), device=self.device)
        for w in weight_list:
            if isinstance(w, tuple):
                l2_reg = torch.norm(w[1], p=self.p, )
            else:
                l2_reg = torch.norm(w, p=self.p, )
            reg_loss = reg_loss + l2_reg
        return self.weight_decay * reg_loss