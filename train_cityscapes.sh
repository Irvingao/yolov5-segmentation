CUDA_VISIBLE_DEVICES=0 
# rm -rf log_dir run
python scripts/train_seg.py --batch-size 8 --epochs 300 --no-val --eval-interval 1 --seg --dataset cityscapes 
# --weights 'pretrained_models/model_best.pth.tar' 