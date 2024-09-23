import numpy as np

y_sig = np.loadtxt("y_sig.csv", delimiter=",")
y_sig_np = np.loadtxt("y_sig_np.csv", delimiter=",")
y_pred_1_np = np.loadtxt("y_pred_1_np.csv", delimiter=",")

diff = np.where(y_sig != y_sig_np)
print(diff)
print(y_sig[diff])
print(y_sig_np[diff])

# 找出A中大于1或小于0的位置
y_sig_condition = (y_sig > 1) | (y_sig < 0)
out_of_range_indices = np.where(y_sig_condition)
print("y_sig中大于1或小于0的位置:", out_of_range_indices)

y_sig_np_condition = (y_sig_np > 1) | (y_sig_np < 0)
out_of_range_indices = np.where(y_sig_np_condition)
print("y_sig_np中大于1或小于0的位置:", out_of_range_indices)

print(y_pred_1_np[662])