import torch
import numpy as np
import os
from tqdm import tqdm
import sys
import logging


def data_filter(train_gen, valid_gen, test_gen, M, data_threshold=0.9, **params):
    M.filter.data_filter = True

    for mode, data_gen in zip(["train", "valid", "test"], [train_gen, valid_gen, test_gen]):
        data_name = mode + "_data"
        data_path = params[data_name]
        filtered_data_path = os.path.join(os.path.dirname(data_path), f"filtered_v2_{mode}.npy")

        # var_sim_scores = []
        # data_iter = tqdm(data_gen, disable=False, file=sys.stdout)
        # for batch_data in data_iter:
        #     # calculate sim_score
        #     with torch.no_grad():
        #         X = M.get_inputs(batch_data)
        #         feature_emb_dict = M.embedding_layer(X)
        #         sim_score = M.filter(X, feature_emb_dict, -1)

        #         sim_mask = (sim_score != 0)
        #         means_sim_score = sim_score.sum(dim=1) / sim_mask.sum(dim=1)
        #         var_sim_score = ((sim_mask * (sim_score - (means_sim_score.unsqueeze(1))) ** 2).sum(dim=1) / sim_mask.sum(dim=1)).detach().cpu().numpy()
        #         var_sim_scores.extend(var_sim_score)
    
        # -------------------------- 根据每条数据的方差进行采样 --------------------------
        # var_sim_scores = np.argsort(-np.array(var_sim_scores))
        # if data_threshold < 1:
        #     threshold = int(var_sim_scores.shape[0] * data_threshold)
        # else:
        #     threshold = data_threshold
        # var_sim_scores = np.sort(var_sim_scores[:threshold])

        # means_sim_scores = []
        # data_iter = tqdm(data_gen, disable=False, file=sys.stdout)
        # for batch_data in data_iter:
        #     with torch.no_grad():
        #         X = M.get_inputs(batch_data)
        #         feature_emb_dict = M.embedding_layer(X)
        #         sim_score = M.filter(X, feature_emb_dict, -1)
        #         sim_mask = (sim_score != 0)
        #         means_sim_score = sim_score.sum(dim=1) / sim_mask.sum(dim=1)
        #         means_sim_scores.extend(means_sim_score.detach().cpu().numpy())

        # means_sim_scores = np.array(means_sim_scores)
        # dist_from_mean = np.abs(means_sim_scores - np.mean(means_sim_scores))
        # dist_from_mean /= np.max(dist_from_mean)
        # sampling_probs = dist_from_mean ** 3

        # -------------------------- 分段采样 --------------------------
        # if data_threshold < 1:
        #     threshold = int(means_sim_scores.shape[0] * data_threshold)
        # else:
        #     threshold = data_threshold
        # dist_from_mean = np.argsort(dist_from_mean)
        # slice_point = 9 / 16  # 从靠近均值前 slice_point 的数据中随机采样, 补齐 threshold
        # filtered_data_idx = np.sort(np.concatenate((np.random.choice(dist_from_mean[:int(slice_point * dist_from_mean.shape[0])], 
        #                                                              size=threshold - int((1 - slice_point) * dist_from_mean.shape[0]), replace=False), 
        #                                             dist_from_mean[int(slice_point * dist_from_mean.shape[0]):])))

        # 

        # sampling_probs /= np.sum(sampling_probs)

        filtered_data_idx = []

        bucket_len = 0.004
        bucket_bins = np.arange(0.7, 1 + bucket_len, bucket_len)
        bucket = np.zeros(bucket_bins.shape[0] - 1)
        bucket_up_bound = {"train": 1e6, "valid": 1e5, "test": 1e5}

        data_iter = tqdm(data_gen, disable=False, file=sys.stdout)
        for batch_idx, batch_data in enumerate(data_iter):
            with torch.no_grad():
                X = M.get_inputs(batch_data)
                feature_emb_dict = M.embedding_layer(X)
                sim_score = M.filter(X, feature_emb_dict, -1)

            for data_idx, line in enumerate(sim_score.detach().cpu().numpy()):
                if np.max(bucket) > bucket_up_bound[mode]:
                    continue
                filtered_line = [x for x in line if x != 0]
                counts, _ = np.histogram(filtered_line, bins=bucket_bins)
                bucket += counts
                filtered_data_idx.append(data_idx + batch_idx * sim_score.shape[0])

            # print(f"TRAIN DATA TEST: filtered_data_idx len: {len(filtered_data_idx)}")

        original_data = np.load(os.path.join(os.path.dirname(data_path), f"{mode}.npy"))
        # filtered_data = original_data[var_sim_scores]
        # if data_threshold < 1:
        #     threshold = int(means_sim_scores.shape[0] * data_threshold)
        # else:
        #     threshold = data_threshold
        # filtered_data_idx = np.sort(np.random.choice(np.arange(original_data.shape[0]), size=threshold, replace=False, p=sampling_probs))
        filtered_data = original_data[filtered_data_idx]
        logging.info(f"=========={data_name} original data shape: {original_data.shape}==========")
        logging.info(f"=========={data_name} filtered data shape: {filtered_data.shape}==========")
        np.save(filtered_data_path, filtered_data)
        params[data_name] = os.path.join(os.path.dirname(data_path), f"filtered_v2_{mode}.parquet")
        logging.info(f"=========={data_name} filterd data saved to {filtered_data_path}==========")

    M.filter.data_filter = False

    return params["train_data"], params["valid_data"], params["test_data"]

