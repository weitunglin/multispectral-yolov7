# run single-modality-rgb
```
python3 train.py --weights yolov7_training.pt --cfg cfg/training/yolov7-custom.yaml --data data/custom-single-modality.yaml --epochs 200 --cache-images --v5-metric --name single-modality-rgb --workers 4 --batch-size 8
```

# run single-modality-dvs
```
python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 9528 train.py --workers 8 --device 2,3 --sync-bn --batch-size 32 --data data/custom-single-modality-dvs.yaml --img 640 640 --cfg cfg/training/yolov7-custom-single-modality.yaml --weights yolov7_training.pt --name single-modality-dvs --hyp data/hyp.scratch.p5.yaml --epochs 200
```