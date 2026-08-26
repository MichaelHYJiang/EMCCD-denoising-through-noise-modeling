export CUDA_VISIBLE_DEVICES=0,1,2,3;
python ./train/train_denoise_cell.py --arch Uformer_B --batch_size 4 --gpu '2,3' \
    --train_ps 512 --train_dir ../2021/cell2021/ --env _0113-EMCCD-ratio_1-20-limited-sigma-cell_finetune \
    --input_dir preprocessed_input --gt_dir new_FPN_removed_GT \
    --dd_in 1 --in_chans 1 \
    --val_dir ../2021/testset/ --save_dir ./logs/ \
    --dataset mono-SID-cell2021 --warmup \
    --nepoch 2000 --noise_model EMCCD --resume \
     --pretrain_weights ./logs/denoising/mono-SID/Uformer_B_0110-EMCCD-ratio_1-20-limited-sigma/models/model_latest.pth
