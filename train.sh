python train.py --name split8 \
    --hdf5FolderPath D:/Dataset/FAIR-Play/splits/split8 \
    --save_epoch_freq 10 \
    --display_freq 10 \
    --save_latest_freq 10 \
    --batchSize 16 \
    --learning_rate_decrease_itr 10 \
    --niter 10 \
    --lr_visual 0.0001 \
    --lr_audio 0.001 \
    --nThreads 1 \
    --gpu_ids 0 \
    --validation_on \
    --validation_freq 10 \
    --validation_batches 50 \
    --stereo_loss_weight 44 \
    --val_return_key stereo_loss_fusion \
    --tensorboard True \
    |& tee -a logs/split8.log
