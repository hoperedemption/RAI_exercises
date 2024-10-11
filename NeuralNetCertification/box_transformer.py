import torch
import torch.nn as nn

# Implementation on an Abstract Box Transformer for Neural Network Certification

class AbstractBox:

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb
        self.ub = ub

    # TODO: define the following methods ...

    # we're still working with images so after each operation, make sure that the individual
    # elements of the vectors are within the range [0,1]
    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> 'AbstractBox':
        lower_bound = (x - eps).clamp_(min=0, max=1)
        upper_bound = (x + eps).clamp_(min=0, max=1)
        return AbstractBox(lower_bound, upper_bound)

    # clearly the biggest possible x can be ub --> biggest possible normalised x is ub/norm(ub)
    def propagate_normalize(self, normalize: Normalize) -> 'AbstractBox':
        lower_bound = normalize(self.lb)
        upper_bound = normalize(self.ub)
        return AbstractBox(lower_bound, upper_bound)

    def propagate_view(self, view: View) -> 'AbstractBox':
        lower_bound = view(self.lb)
        upper_bound = view(self.ub)
        return AbstractBox(lower_bound, upper_bound)

    # ok yeah so not as simple bc xA^T can switch the inequalities if A^T has negative entries!
    def propagate_linear(self, fc: nn.Linear) -> 'AbstractBox':
        A = fc.weight.transpose(0, 1) # the entries in the matrix
        b = fc.bias # the bias vector
        # for upper bound select entries from ub if the correspoding a entry is > 0, and from lb otherwise (<= 0)

        lb_expanded = self.lb.transpose(0, 1).repeat(1, A.shape[1])
        ub_expanded = self.ub.transpose(0, 1).repeat(1, A.shape[1])
        mixed_box_lower = torch.where(A >= 0, lb_expanded * A, ub_expanded * A).sum(dim=0) + b
        mixed_box_upper = torch.where(A >= 0, ub_expanded * A, lb_expanded * A).sum(dim=0) + b

        mixed_box_lower = mixed_box_lower.unsqueeze(-1).transpose(0, 1)
        mixed_box_upper = mixed_box_upper.unsqueeze(-1).transpose(0, 1)
        return AbstractBox(mixed_box_lower, mixed_box_upper)

    def propagate_relu(self, relu: nn.ReLU) -> 'AbstractBox':
        lower_bound = relu(self.lb)
        upper_bound = relu(self.ub)
        return AbstractBox(lower_bound, upper_bound)

    def check_postcondition(self, y) -> bool:
        # check that all inputs in given range classify to y
        # thus we want to test whether there exist a li in the box that dominates all other uj
        assert self.lb.shape[0] == self.ub.shape[0]
        c = self.lb.shape[1] # number of classes
        lb_y = self.lb[..., y]
        mask = (self.ub[0] < lb_y)
        return bool((mask.sum() == 9).item())
