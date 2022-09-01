# Clusformer: A Transformer based Clustering Approach to Unsupervised Large-scale Face and Visual Landmark Recognition

This repo is an official implementations of [Clusformer](https://openaccess.thecvf.com/content/CVPR2021/papers/Nguyen_Clusformer_A_Transformer_Based_Clustering_Approach_to_Unsupervised_Large-Scale_Face_CVPR_2021_paper.pdf)

## Build knns 
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

## Train clusformer 
```
python -u -m torch.distributed.launch --nproc_per_node=4 main.py    --output_dir ./logs/baseline/ \
                                                                    --batch_size_per_gpu 512
```

## Citation
If you find this repository useful, please consider giving a star :star: and citation
```
@INPROCEEDINGS{nguyen2021clusformer,
  author={Nguyen, Xuan-Bac and Bui, Duc Toan and Duong, Chi Nhan and Bui, Tien D. and Luu, Khoa},
  booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Clusformer: A Transformer based Clustering Approach to Unsupervised Large-scale Face and Visual Landmark Recognition}, 
  year={2021},
  pages={10842-10851},
}
```
