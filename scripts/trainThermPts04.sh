#!/usr/bin/env bash
set -e

source ~/envs/y7_1/bin/activate

# export CUDA_VISIBLE_DEVICES=0

python3 train.py \
  --epochs 300 \
  --workers 16 \
  --device 0 \
  --batch-size 48 \
  --save_period 5 \
  --data data/pts04Therm.lin.yaml \
  --img-size 736 960 \
  --cfg cfg/training/yolov7.yaml \
  --weights models/yolov7.pt \
  --name Vis20251105_pts04-lin-dbg \
  --exist-ok \
  --hyp data/hyp.scratch.p5.hyp1.yaml
