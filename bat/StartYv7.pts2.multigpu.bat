python train.py --workers 8 --device 0,1 --batch-size 4 --save_period 5 --data .\data\PtsCoco_filt.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'D:\Local\Yolo\Yv7\yolov7\models\yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.pts2.yaml

