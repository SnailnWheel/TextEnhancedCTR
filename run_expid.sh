#!/bin/bash

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_kkbox_x1_id_only
echo "-------------DCN_kkbox_x1_id_only DONE---------------"

mv model_zoo/DCN/DCN_torch/checkpoints/DCN_kkbox_x1/KKBox_x1_id_only/user_id_pretrained_emb.parquet data/KKBox/user_id_pretrained_emb_DCN.parquet
mv model_zoo/DCN/DCN_torch/checkpoints/DCN_kkbox_x1/KKBox_x1_id_only/item_id_pretrained_emb.parquet data/KKBox/item_id_pretrained_emb_DCN.parquet
mv model_zoo/DCN/DCN_torch/checkpoints/DCN_kkbox_x1/KKBox_x1_id_only/cate_id_pretrained_emb.parquet data/KKBox/cate_id_pretrained_emb_DCN.parquet
echo "-------------DCN_kkbox_x1 id embedding moving DONE---------------"

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_kkbox_x1_filter_cosine_v2
echo "-------------DCN_kkbox_x1_filter_cosine_v2 DONE---------------"

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_kkbox_x1_filter_cosine_pre_v2
echo "-------------DCN_kkbox_x1_filter_cosine_pre_v2 DONE---------------"

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_kkbox_x1_filter_attn_score_v2
echo "-------------DCN_kkbox_x1_filter_attn_score_v2 DONE---------------"

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_kkbox_x1_filter_attn_score_pre_v2
echo "-------------DCN_kkbox_x1_filter_attn_score_pre_v2 DONE---------------"

echo "=============DCN experiments DONE================"


python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_kkbox_x1_id_only
echo "-------------WideDeep_kkbox_x1_id_only DONE---------------"

mv model_zoo/WideDeep/WideDeep_torch/checkpoints/WideDeep_kkbox_x1/KKBox_x1_id_only/user_id_pretrained_emb.parquet data/KKBox/user_id_pretrained_emb_WideDeep.parquet
mv model_zoo/WideDeep/WideDeep_torch/checkpoints/WideDeep_kkbox_x1/KKBox_x1_id_only/item_id_pretrained_emb.parquet data/KKBox/item_id_pretrained_emb_WideDeep.parquet
mv model_zoo/WideDeep/WideDeep_torch/checkpoints/WideDeep_kkbox_x1/KKBox_x1_id_only/cate_id_pretrained_emb.parquet data/KKBox/cate_id_pretrained_emb_WideDeep.parquet
echo "-------------WideDeep_kkbox_x1 id embedding moving DONE---------------"

python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_kkbox_x1_filter_cosine_v2
echo "-------------WideDeep_kkbox_x1_filter_cosine_v2 DONE---------------"

python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_kkbox_x1_filter_cosine_pre_v2
echo "-------------WideDeep_kkbox_x1_filter_cosine_pre_v2 DONE---------------"

python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_kkbox_x1_filter_attn_score_v2
echo "-------------WideDeep_kkbox_x1_filter_attn_score_v2 DONE---------------"

python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_kkbox_x1_filter_attn_score_pre_v2
echo "-------------WideDeep_kkbox_x1_filter_attn_score_pre_v2 DONE---------------"

echo "=============WideDeep experiments DONE================"


python model_zoo/PNN/run_expid.py --expid PNN_kkbox_x1_id_only
echo "-------------PNN_kkbox_x1_id_only DONE---------------"

mv model_zoo/PNN/checkpoints/PNN_kkbox_x1/KKBox_x1_id_only/user_id_pretrained_emb.parquet data/KKBox/user_id_pretrained_emb_PNN.parquet
mv model_zoo/PNN/checkpoints/PNN_kkbox_x1/KKBox_x1_id_only/item_id_pretrained_emb.parquet data/KKBox/item_id_pretrained_emb_PNN.parquet
mv model_zoo/PNN/checkpoints/PNN_kkbox_x1/KKBox_x1_id_only/cate_id_pretrained_emb.parquet data/KKBox/cate_id_pretrained_emb_PNN.parquet
echo "-------------PNN_kkbox_x1 id embedding moving DONE---------------"

python model_zoo/PNN/run_expid.py --expid PNN_kkbox_x1_filter_cosine_v2
echo "-------------PNN_kkbox_x1_filter_cosine_v2 DONE---------------"

python model_zoo/PNN/run_expid.py --expid PNN_kkbox_x1_filter_cosine_pre_v2
echo "-------------PNN_kkbox_x1_filter_cosine_pre_v2 DONE---------------"

python model_zoo/PNN/run_expid.py --expid PNN_kkbox_x1_filter_attn_score_v2
echo "-------------PNN_kkbox_x1_filter_attn_score_v2 DONE---------------"

python model_zoo/PNN/run_expid.py --expid PNN_kkbox_x1_filter_attn_score_pre_v2
echo "-------------PNN_kkbox_x1_filter_attn_score_pre_v2 DONE---------------"

echo "=============PNN experiments DONE================"


python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_amazonmovies_x1_filter_cosine_pre_v2
echo "-------------DCN_amazonmovies_x1_filter_cosine_pre_v2 DONE---------------"

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_amazonmovies_x1_filter_attn_score_pre_v2
echo "-------------DCN_amazonmovies_x1_filter_attn_score_pre_v2 DONE---------------"

python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_amazonmovies_x1_filter_cosine_pre_v2
echo "-------------WideDeep_amazonmovies_x1_filter_cosine_pre_v2 DONE---------------"

python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_amazonmovies_x1_filter_attn_score_pre_v2
echo "-------------WideDeep_amazonmovies_x1_filter_attn_score_pre_v2 DONE---------------"

python model_zoo/PNN/run_expid.py --expid PNN_amazonmovies_x1_filter_cosine_pre_v2
echo "-------------PNN_amazonmovies_x1_filter_cosine_pre_v2 DONE---------------"

python model_zoo/PNN/run_expid.py --expid PNN_amazonmovies_x1_filter_attn_score_pre_v2
echo "-------------PNN_amazonmovies_x1_filter_attn_score_pre_v2 DONE---------------"
