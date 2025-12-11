python train.py --workers 8 --device 0 --batch-size 16 --save_period 10 --data .\data\PtsCoco_filt.yaml --img 640 640 --cfg cfg/training/yolov7x.yaml  --name yolov7xSubsetDl2 --hyp data/hyp.scratch.p5.yaml --weights ./models/yolov7x.pt

