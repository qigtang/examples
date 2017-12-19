import torch
from torch.autograd import Variable

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import torch.distributed as dist

from torch.nn.modules import Module

class DistributedDataParallel(Module):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DistributedDataParallel, self).__init__()

        if device_ids is None or len(device_ids)>1:# or dist._backend != dist.dist_backend.NCCL:
            raise RuntimeError("This version of DistributedDataParallel requires NCCL as backend and a single GPU/process.")
        if output_device is None:
            output_device = device_ids[0]
            
        self.dim = dim
        self.module = module
        self.device_ids = device_ids

        #self._nccl_stream = torch.cuda.Stream()

        for p in self.module.state_dict().values():
            dist.broadcast(p, 0)
            
        for param in list(module.parameters()):
            dist.broadcast(param.data, 0)

        def test():
            if(self.flag):
                self.flag=False
                buckets = {}
                for param in self.module.parameters():
                    if param.requires_grad:
                        tp = type(param.data)
                        if tp not in buckets:
                            buckets[tp]=[]
                        buckets[tp].append(param)
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)
                        
            
        for param in list(self.module.parameters()):
            def hook(*unused):
                param._execution_engine.queue_callback(test)
            param.register_hook(hook)


    def forward(self, *inputs, **kwargs):
        self.flag = True

        return self.module(*inputs, **kwargs)

    '''
    def _sync_buffers(self):
        buffers = list(self.module._all_buffers())
        if len(buffers) > 0:
            # cross-node buffer sync
            flat_buffers = _flatten_dense_tensors(buffers)
            dist.broadcast(flat_buffers, 0)
            for buf, synced in zip(buffers, _unflatten_dense_tensors(flat_buffers, buffers)):
                buf.copy_(synced)
    def train(self, mode=True):
        # Clear NCCL communicator and CUDA event cache of the default group ID,
        # These cache will be recreated at the later call. This is currently a
        # work-around for a potential NCCL deadlock.
        if dist._backend == dist.dist_backend.NCCL:
            dist._clear_group_cache()
        super(DistributedDataParallel, self).train(mode)
        self.module.train(mode)
    '''
