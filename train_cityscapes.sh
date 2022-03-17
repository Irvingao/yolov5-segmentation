CUDA_VISIBLE_DEVICES=0 
set PYTHONPATH="/root/ghz_ws/flexible-yolov5:$PYTHONPATH"
rm -rf log_dir run
python scripts/train_seg.py --batch-size 8 --epochs 200 --eval-interval 5 --seg --dataset cityscapes 
# --weights 'pretrained_models/model_best.pth.tar' 