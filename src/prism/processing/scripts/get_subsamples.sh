python subsample.py \
  --out_dir /path/to/vtknpy_out \
  --target_N 1000000 \
  --bbox_min -550 -550 0 \
  --bbox_max  550  550 1100 \
  --decay_xy 0.2 \
  --decay_z 0.2 \
  --suffix 1M \
  --jobs 8 \
  --backend multiprocessing
