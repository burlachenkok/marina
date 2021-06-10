import utils
import numpy as np
import sys

if len(sys.argv) <= 1:
    print("Launch as: show.py <serialized_file.bin>")
    sys.exit(-1)

d = utils.deserialize(sys.argv[1])
print("KEYS")
print("=======================================================")
for k in d.keys():   
    print(k)
print("=======================================================")
print("DESCR")
print(d["descr"])
print("=======================================================")
print("COPMPRESSOR")
print(d["descr"]["init_compressor"])
print("=======================================================")
print("DATASET")
print("name:", d["dataset"])
print("samples:", d["train_set_size"])
print("d:", d["train_variables"])
print("fi_grad_calcs_by_nodes shape:", d["fi_grad_calcs_by_nodes"].shape)
print("transfered_bits_by_nodes shape:", d["transfered_bits_by_nodes"].shape)

#print("NODES SHAPE")
#print(d["xk_nodes"].shape)
#print("SAMPLED NODES SHAPE")
#print(d["xk_nodes_sampled"].shape)
#print("=======================================================")
