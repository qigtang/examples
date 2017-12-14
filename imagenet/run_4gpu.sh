#!/bin/bash

for GPU in `seq 0 2`; do python main.py -j 10 --arch resnet50 --b 128 --world-size 4 --gpu $GPU --dist-backend nccl --fp16 . > ${GPU}.log & done ; python main.py --arch resnet50 --b 128 --world-size 4 --gpu 3 -j 10 --dist-backend nccl --fp16 .
