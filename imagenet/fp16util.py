import torch
import torch.nn as nn


class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def copy_in_params(net, params):
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)


def set_grad(params, params_with_grad):

    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))
        param.grad.data.copy_(param_w_grad.grad.data)


#BatchNorm layers to have parameters in single precision.
#Find all layers and convert them back to float. This can't
#be done with built in .apply as that function will apply
#fn to all modules, parameters, and buffers. Thus we wouldn't
#be able to guard the float conversion based on the module type.
def BN_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module

def network_to_half(network):
    return nn.Sequential(tofp16(), BN_convert_float(network.cuda().half()))


class prefetch():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
        
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)
            torch.cuda.current_stream().wait_stream(self.stream)
            
    def next(self):
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
