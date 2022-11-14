# run single-modality-rgb
```
python3 train.py --weights yolov7_training.pt --cfg cfg/training/yolov7-custom.yaml --data data/custom-single-modality.yaml --epochs 200 --cache-images --v5-metric --name single-modality-rgb --workers 4 --batch-size 8
```