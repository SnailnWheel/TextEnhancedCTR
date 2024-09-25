#!/bin/bash

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_yelp_x1_id_only --gpu 1
echo "-------------DCN_yelp_x1_id_only DONE---------------"

mv model_zoo/DCN/DCN_torch/checkpoints/DCN_yelp_x1/Yelp_x1_id_only/user_id_pretrained_emb.parquet data/Yelp/user_id_pretrained_emb_DCN.parquet
mv model_zoo/DCN/DCN_torch/checkpoints/DCN_yelp_x1/Yelp_x1_id_only/item_id_pretrained_emb.parquet data/Yelp/item_id_pretrained_emb_DCN.parquet
mv model_zoo/DCN/DCN_torch/checkpoints/DCN_yelp_x1/Yelp_x1_id_only/cate_id_pretrained_emb.parquet data/Yelp/cate_id_pretrained_emb_DCN.parquet
echo "-------------DCN_yelp_x1 id embedding moving DONE---------------"

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_yelp_x1_filter_cosine_v2 --gpu 1
echo "-------------DCN_yelp_x1_filter_cosine_v2 DONE---------------"

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_yelp_x1_filter_cosine_pre_v2 --gpu 1
echo "-------------DCN_yelp_x1_filter_cosine_pre_v2 DONE---------------"

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_yelp_x1_filter_attn_score_v2 --gpu 1
echo "-------------DCN_yelp_x1_filter_attn_score_v2 DONE---------------"

python model_zoo/DCN/DCN_torch/run_expid.py --expid DCN_yelp_x1_filter_attn_score_pre_v2 --gpu 1
echo "-------------DCN_yelp_x1_filter_attn_score_pre_v2 DONE---------------"

echo "=============DCN experiments DONE================"


python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_yelp_x1_id_only --gpu 1
echo "-------------WideDeep_yelp_x1_id_only DONE---------------"

mv model_zoo/WideDeep/WideDeep_torch/checkpoints/WideDeep_yelp_x1/Yelp_x1_id_only/user_id_pretrained_emb.parquet data/Yelp/user_id_pretrained_emb_WideDeep.parquet
mv model_zoo/WideDeep/WideDeep_torch/checkpoints/WideDeep_yelp_x1/Yelp_x1_id_only/item_id_pretrained_emb.parquet data/Yelp/item_id_pretrained_emb_WideDeep.parquet
mv model_zoo/WideDeep/WideDeep_torch/checkpoints/WideDeep_yelp_x1/Yelp_x1_id_only/cate_id_pretrained_emb.parquet data/Yelp/cate_id_pretrained_emb_WideDeep.parquet
echo "-------------WideDeep_yelp_x1 id embedding moving DONE---------------"

python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_yelp_x1_filter_cosine_v2 --gpu 1
echo "-------------WideDeep_yelp_x1_filter_cosine_v2 DONE---------------"

python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_yelp_x1_filter_cosine_pre_v2 --gpu 1
echo "-------------WideDeep_yelp_x1_filter_cosine_pre_v2 DONE---------------"

python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_yelp_x1_filter_attn_score_v2 --gpu 1
echo "-------------WideDeep_yelp_x1_filter_attn_score_v2 DONE---------------"

python model_zoo/WideDeep/WideDeep_torch/run_expid.py --expid WideDeep_yelp_x1_filter_attn_score_pre_v2 --gpu 1
echo "-------------WideDeep_yelp_x1_filter_attn_score_pre_v2 DONE---------------"

echo "=============WideDeep experiments DONE================"


python model_zoo/PNN/run_expid.py --expid PNN_yelp_x1_id_only --gpu 1
echo "-------------PNN_yelp_x1_id_only DONE---------------"

mv model_zoo/PNN/checkpoints/PNN_yelp_x1/Yelp_x1_id_only/user_id_pretrained_emb.parquet data/Yelp/user_id_pretrained_emb_PNN.parquet
mv model_zoo/PNN/checkpoints/PNN_yelp_x1/Yelp_x1_id_only/item_id_pretrained_emb.parquet data/Yelp/item_id_pretrained_emb_PNN.parquet
mv model_zoo/PNN/checkpoints/PNN_yelp_x1/Yelp_x1_id_only/cate_id_pretrained_emb.parquet data/Yelp/cate_id_pretrained_emb_PNN.parquet
echo "-------------PNN_yelp_x1 id embedding moving DONE---------------"

python model_zoo/PNN/run_expid.py --expid PNN_yelp_x1_filter_cosine_v2 --gpu 1
echo "-------------PNN_yelp_x1_filter_cosine_v2 DONE---------------"

python model_zoo/PNN/run_expid.py --expid PNN_yelp_x1_filter_cosine_pre_v2 --gpu 1
echo "-------------PNN_yelp_x1_filter_cosine_pre_v2 DONE---------------"

python model_zoo/PNN/run_expid.py --expid PNN_yelp_x1_filter_attn_score_v2 --gpu 1
echo "-------------PNN_yelp_x1_filter_attn_score_v2 DONE---------------"

python model_zoo/PNN/run_expid.py --expid PNN_yelp_x1_filter_attn_score_pre_v2 --gpu 1
echo "-------------PNN_yelp_x1_filter_attn_score_pre_v2 DONE---------------"

echo "=============PNN experiments DONE================"

