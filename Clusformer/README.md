# Build knns 
```
for fold in 1 3 5 7 9; do
    feature_path=/home/data/clustering/data/features/part${fold}_test.bin
    label_path=/home/data/clustering/data/labels/part${fold}_test.meta
    knn_method=faiss_gpu
    out_dir=/home/data/clustering/data_new/knns/part${fold}_test/
    k=256
    feature_dim=256
    is_rebuild=True

    python build_knn.py     --feature_path ${feature_path} \
                            --label_path ${label_path} \
                            --knn_method ${knn_method} \
                            --out_dir ${out_dir} \
                            --k ${k} \
                            --is_rebuild ${is_rebuild} \
                            --feature_dim ${feature_dim}
done
```

# Train clusformer 
```
python -u -m torch.distributed.launch --nproc_per_node=4 main.py    --output_dir ./logs/baseline/ \
                                                                    --batch_size_per_gpu 512
```