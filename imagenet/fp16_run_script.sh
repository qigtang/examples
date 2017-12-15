#!/bin/bash

HELP="$(basename "$0") [-h] [--ngpus n] -- Script to run single node multi-gpu imagenet test.

where:
    -h  show this help text
    --ngpus number of GPU's to use. (Starts with GPU 0)"

case "$1" in
    -h)
	echo $HELP
	;;
    --ngpus)
	case $2 in
	    ''|*[!0-9]*) echo "Error: Invaliad option for -ngpus"; exit;;
	    *) NGPUS=$2 ;;
	esac
	;;
    -*)
	echo "Error: Unknown option: $1" >&2
	exit 1
	;;
    *)  echo "Error: Need to specify number of GPUs"
	echo $HELP
	exit 1
	;;
esac

((M_ONE = NGPUS-1))
((M_TWO = NGPUS-2))
((PROCS = 40/NGPUS))

for GPU in `seq 0 $M_TWO`;
do
    python main.py -j $PROCS --arch resnet50 --b 128 --world-size $NGPUS --gpu $GPU --dist-backend nccl --fp16 . > gpu_${GPU}.log &
done

python main.py -j $PROCS --arch resnet50 --b 128 --world-size $NGPUS --gpu $M_ONE --dist-backend nccl --fp16 . | tee -a gpu_${M_ONE}.log

