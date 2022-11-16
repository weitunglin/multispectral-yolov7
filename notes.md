## Useful Commands

### train single-modality-rgb
```
python3 train.py --weights yolov7_training.pt --cfg cfg/training/yolov7-custom.yaml --data data/custom-single-modality.yaml --epochs 200 --cache-images --v5-metric --name single-modality-rgb --workers 4 --batch-size 8
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 9528 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 48 --data data/custom-single-modality-rgb.yaml --img 640 640 --cfg cfg/training/yolov7-custom-single-modality.yaml --weights yolov7_training.pt --name single-modality-rgb --hyp data/hyp.scratch.p5.yaml --v5-metric --epochs 200
```

### test single-modality-rgb
```
python3 test.py --data data/custom-single-modality-rgb.yaml --img 640 --batch 32 --conf 0.65 --iou 0.65 --device 0 --weights runs/train/single-modality-rgb14/weights/best.pt --name single-modality-rgb --no-trace
```

### run single-modality-dvs
```
python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 9528 train.py --workers 8 --device 2,3 --sync-bn --batch-size 32 --data data/custom-single-modality-dvs.yaml --img 640 640 --cfg cfg/training/yolov7-custom-single-modality.yaml --weights yolov7_training.pt --name single-modality-dvs --hyp data/hyp.scratch.p5.yaml --v5-metric --epochs 200
```

### test single-modality-dvs
```
python3 test.py --data data/custom-single-modality-dvs.yaml --img 640 --batch 32 --conf 0.65 --iou 0.65 --device 0 --weights runs/train/single-modality-dvs13/weights/best.pt --name single-modality-dvs --no-trace
```

### train single-modality-radar
```
python3 -m torch.distributed.launch --nproc_per_node 3 --master_port 9528 train.py --workers 8 --device 0,1,3 --sync-bn --batch-size 48 --data data/custom-single-modality-radar.yaml --img 640 640 --cfg cfg/training/yolov7-custom-single-modality.yaml --weights yolov7_training.pt --name single-modality-radar --hyp data/hyp.scratch.p5.yaml --v5-metric --epochs 200
```

### test single-modality-radar
```
python3 test.py --data data/custom-single-modality-radar.yaml --img 640 --batch 32 --conf 0.65 --iou 0.65 --device 0 --weights runs/train/single-modality-radar2/weights/best.pt --name single-modality-radar --no-trace
```

### run multi-modality
- without pre-trained
```
python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 9528 train.py --workers 8 --device 2,3 --sync-bn --batch-size 12 --data data/custom-multi-modality.yaml --img 640 640 --cfg cfg/training/yolov7-two-stream-transformer.yaml --weights '' --name two-stream-transformer --hyp data/hyp.scratch.multi-modality.yaml --v5-metric --epochs 200 --two-stream
```

- with yolov7_training.pt pre-trained backbone
```
python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 9528 train.py --workers 8 --device 2,3 --sync-bn --batch-size 12 --data data/custom-multi-modality.yaml --img 640 640 --cfg cfg/training/yolov7-two-stream-transformer.yaml --weights yolov7_training.py --name two-stream-transformer-fine-tune --hyp data/hyp.scratch.multi-modality.yaml --v5-metric --epochs 200 --two-stream
```

### test multi-modality
```
python3 test.py --data data/custom-multi-modality.yaml --img 640 --batch 32 --conf 0.65 --iou 0.65 --device 0 --weights runs/train/two-stream-transformer-fine-tune8/weights/best.pt --name two-stream-transformer-fine-tune --two-stream --no-trace
```

## Results

### multi-modality two-stream transfomer

| Class          | Images | Labels | P     | R     | MAPE   | mAP@.5 | mAP@.5:.95 |
| :------------- | :----- | :----- | :---- | :---- | :----- | :----- | :--------- |
| all            | 827    | 6963   | 0.951 | 0.596 | 0.0493 | 0.595  | 0.442      |
| pedestrain     | 827    | 1845   | 0.931 | 0.471 | 0.0686 | 0.469  | 0.282      |
| bicycle        | 827    | 176    | 0.93  | 0.528 | 0.07   | 0.526  | 0.364      |
| motorcycle     | 827    | 75     | 0.98  | 0.64  | 0.0204 | 0.644  | 0.434      |
| car            | 827    | 2138   | 0.956 | 0.648 | 0.0435 | 0.648  | 0.516      |
| bus            | 827    | 126    | 0.957 | 0.714 | 0.0426 | 0.712  | 0.597      |
| truck          | 827    | 569    | 0.944 | 0.591 | 0.0562 | 0.59   | 0.474      |
| construction   | 827    | 117    | 0.975 | 0.667 | 0.025  | 0.668  | 0.516      |
| movable_object | 827    | 1917   | 0.932 | 0.513 | 0.0682 | 0.508  | 0.353      |

Speed: 51.5/1.1/52.6 ms inference/NMS/total per 640x640 image at batch-size 1

### single-modality-dvs
| Class          | Images | Labels | P     | R     | MAPE   | mAP@.5 | mAP@.5:.95 |
| :------------- | :----- | :----- | :---- | :---- | :----- | :----- | :--------- |
| all            | 827    | 6963   | 0.938 | 0.737 | 0.0623 | 0.732  | 0.548      |
| pedestrian     | 827    | 1845   | 0.934 | 0.531 | 0.0658 | 0.527  | 0.322      |
| bicycle        | 827    | 176    | 0.934 | 0.648 | 0.0656 | 0.644  | 0.422      |
| motorcycle     | 827    | 75     | 0.915 | 0.72  | 0.0847 | 0.707  | 0.491      |
| car            | 827    | 2138   | 0.936 | 0.804 | 0.0642 | 0.8    | 0.641      |
| bus            | 827    | 126    | 0.958 | 0.905 | 0.042  | 0.908  | 0.769      |
| truck          | 827    | 569    | 0.954 | 0.831 | 0.0464 | 0.83   | 0.673      |
| construction   | 827    | 117    | 0.941 | 0.821 | 0.0588 | 0.816  | 0.635      |
| movable_object | 827    | 1917   | 0.929 | 0.632 | 0.0713 | 0.627  | 0.435      |

Speed: 22.7/1.0/23.8 ms inference/NMS/total per 640x640 image at batch-size 1

### single-modality-radar
| Class          | Images | Labels | P     | R     | MAPE   | mAP@.5 | mAP@.5:.95 |
| :------------- | :----- | :----- | :---- | :---- | :----- | :----- | :--------- |
| all            | 827    | 6963   | 0.935 | 0.737 | 0.0654 | 0.733  | 0.553      |
| pedestrian     | 827    | 1845   | 0.932 | 0.513 | 0.0679 | 0.512  | 0.309      |
| bicycle        | 827    | 176    | 0.92  | 0.653 | 0.08   | 0.642  | 0.418      |
| motorcycle     | 827    | 75     | 0.914 | 0.707 | 0.0862 | 0.699  | 0.51       |
| car            | 827    | 2138   | 0.94  | 0.796 | 0.0597 | 0.793  | 0.633      |
| bus            | 827    | 126    | 0.934 | 0.897 | 0.0661 | 0.895  | 0.778      |
| truck          | 827    | 569    | 0.964 | 0.842 | 0.0362 | 0.842  | 0.675      |
| construction   | 827    | 117    | 0.953 | 0.863 | 0.0472 | 0.864  | 0.674      |
| movable_object | 827    | 1917   | 0.92  | 0.622 | 0.0802 | 0.617  | 0.428      |

Speed: 22.6/1.0/23.6 ms inference/NMS/total per 640x640 image at batch-size 1