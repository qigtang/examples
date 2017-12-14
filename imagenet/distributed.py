import sys
import math
import threading
import copy

import torch
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl
import torch.distributed as dist

from torch.nn.modules import Module
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply

if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue


class DistributedDataParallel(Module):
    r"""Implements distributed data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. The module is replicated on each machine and each device, and
    each such replica handles a portion of the input. During the backwards
    pass, gradients from each node are averaged.

    The batch size should be larger than the number of GPUs used locally. It
    should also be an integer multiple of the number of GPUs so that each chunk
    is the same size (so that each GPU processes the same number of samples).

    See also: :ref:`distributed-basics` and :ref:`cuda-nn-dataparallel-instead`.
    The same constraints on input as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires the distributed package to be already
    initialized in the process group mode
    (see :func:`torch.distributed.init_process_group`).

    .. warning::
        This module works only with the ``gloo`` backend.

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) is a distributed synchronization
        point. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    .. warning::
        This module assumes all buffers and gradients are dense.

    .. warning::
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast form the module in process of rank
        0, to all other replicas in the system in every iteration.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> torch.distributed.init_process_group(world_size=4, init_method='...')
        >>> net = torch.nn.DistributedDataParallel(model)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DistributedDataParallel, self).__init__()

        if device_ids is None or len(device_ids)>1 or dist._backend != dist.dist_backend.NCCL:
            raise RuntimeError("This version of DistributedDataParallel requires NCCL as backend and a single GPU/process.")
        if output_device is None:
            output_device = device_ids[0]
            
        self.dim = dim
        self.module = module
        self.device_ids = device_ids

        self._nccl_stream = torch.cuda.Stream()

        # Sync params and buffers
        for p in self.module.state_dict().values():
            dist.broadcast(p, 0)

        # Clear NCCL communicator and CUDA event cache of the default group ID,
        # These cache will be recreated at the later call. This is currently a
        # work-around for a potential NCCL deadlock.
        dist._clear_group_cache()


    def forward(self, *inputs, **kwargs):
        #inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        _cuda_inputs = []
        for input in inputs:
            _cuda_inputs.append(input.cuda(self.device_ids[0]))
        _cuda_inputs = tuple(_cuda_inputs)   
        self._sync_buffers()

        
        self.flag = True
        def test():
            if(self.flag):
                buckets = {}
                for param in self.module.parameters():
                    if param.requires_grad:
                        tp = type(p.data)
                        if tp not in buckets:
                            buckets[tp]=[]
                        buckets[tp].append(param)
                for tp, bucket in buckets.iteritems():
                    grads = [param.data.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    nccl.reduce(grads, root=0, streams=nccl_stream)
                    torch.cuda.wait_stream(nccl_stream)
                    for buf, synced in zip(bucket, _unflatten_dense_tensors(grads, bucket)):
                        buf.copy_(synced)
                
                self.flag=False
        
        
        def hook(*unused):
            inputs[0]._execution_engine.queue_callback(test)

        return self.module(*_cuda_inputs, **kwargs)

    def train(self, mode=True):
        # Clear NCCL communicator and CUDA event cache of the default group ID,
        # These cache will be recreated at the later call. This is currently a
        # work-around for a potential NCCL deadlock.
        if dist._backend == dist.dist_backend.NCCL:
            dist._clear_group_cache()
        super(DistributedDataParallel, self).train(mode)
        self.module.train(mode)


    def _sync_buffers(self):
        buffers = list(self.module._all_buffers())
        if len(buffers) > 0:
            # cross-node buffer sync
            flat_buffers = _flatten_dense_tensors(buffers)
            dist.broadcast(flat_buffers, 0)
            for buf, synced in zip(buffers, _unflatten_dense_tensors(flat_buffers, buffers)):
                buf.copy_(synced)
