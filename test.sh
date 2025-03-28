python demo.py --input_audio_length 10 \
    --hop_size 0.1 \
    --use_fusion_pred \
    --weights_visual checkpoints/model/visual_best.pth \
    --weights_audio checkpoints/model/audio_best.pth \
    --weights_fusion checkpoints/model/fusion_best.pth \
    --weights_att1 checkpoints/model/att1_best.pth \
    --output_dir_root eval_demo/stereo/model-best/split5_demo \
    --hdf5FolderPath D:/Dataset/FAIR-Play/splits/split5/test.h5 \
    # --hdf5FolderPath /home/lsj/mono2bin/splits/split1/test.h5
