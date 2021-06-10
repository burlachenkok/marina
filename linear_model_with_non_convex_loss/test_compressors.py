#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Goals of tests:
# - Check compressor behavior
# - Plot compressor behaviour in terms of need to send scalar values over the network

import numpy as np
import matplotlib.pyplot as plt

# Default was 10
plt.rcParams["font.size"] = plt.rcParams["font.size"] + 2

import compressors as cp
import sys, math
import utils

def checkCompressor(c_test, base_vector, KMax, threshold = 0.1):
    average = np.zeros(shape = base_vector.shape);
    processed = 0;
    estimate_second_moment = 0.0

    # Checking that in expectation compressors behaves correctly
    for i in range(KMax):
        result = c_test.compressVector(base_vector)
        l2_norm = np.linalg.norm(result)

        # somehting like rank-one update for the estimation via evaluate mean of:

        # second moment
        estimate_second_moment = (estimate_second_moment * processed + l2_norm * l2_norm) / (processed + 1);

        # first moment
        average = (average * processed + result) / (processed + 1);
        processed = processed + 1;

    base_vector_norm = np.linalg.norm(base_vector)
    rel_error = np.linalg.norm(base_vector - average) / np.linalg.norm(base_vector)

    if rel_error < threshold:
        print("Check that {0} is unbiased: [OK]".format(c_test.name()))
    else:
        print("Check that {0} is unbiased: [FAIL] (relative error is {1})".format(c_test.name(), rel_error))

    if c_test.total_input_components == KMax *D and c_test.really_need_to_send_components <= c_test.total_input_components:
        print("Compressor {0} correctly calculate number of processed items: [OK]".format(c_test.name()))
    else:
        print("Compressor {0} correctly calculate number of processed items: [FAIL]".format(c_test.name()))

    estimated_w = estimate_second_moment/ (base_vector_norm**2) - 1.0    
    print("Compressor {0} has good bound on W: [OK]".format(c_test.name()))
    print("{0}: Estimated W: {1} / Suggested W by method: {2}".format(c_test.name(), estimated_w, c_test.getW()))
    print("")

#============================================================================================================
np.random.seed(2);
D = 15;
KMax = 1000;

base_vector = np.random.uniform(size=(D,1)) * 1000

c1 = cp.Compressor("IdenticalCompressor")
c1.makeIdenticalCompressor()
checkCompressor(c1, base_vector, KMax)

c2 = cp.Compressor("LazyCompressor. P=0.6")
c2.makeLazyCompressor(P = 0.6)
checkCompressor(c2, base_vector, KMax)

c3 = cp.Compressor("LazyCompressor. P=0.5")
c3.makeLazyCompressor(P = 0.5)
checkCompressor(c3, base_vector, KMax)

c4 = cp.Compressor("LazyCompressor. P=0.4")
c4.makeLazyCompressor(P = 0.4)
checkCompressor(c4, base_vector, KMax)

c5 = cp.Compressor(f"RandK. K=8, D={D}")
c5.makeRandKCompressor(K = 8, D = D)
checkCompressor(c5, base_vector, KMax)

c6 = cp.Compressor(f"RandK. K=10, D={D}")
c6.makeRandKCompressor(K = 10, D = D)
checkCompressor(c6, base_vector, KMax)

c7 = cp.Compressor(f"RandK. K=15, D={D}")
c7.makeRandKCompressor(K = 15, D = D)
checkCompressor(c7, base_vector, KMax)

c8 = cp.Compressor("NaturalCompressor for fp64")
c8.makeNaturalCompressorFP64()
checkCompressor(c8, base_vector, KMax)

s = 256
p = 2
c9 = cp.Compressor(f"NaturalDithering for fp64 p={p},s={s}")
c9.makeNaturalDitheringFP64(levels = s, dInput = D, p = p)
checkCompressor(c9, base_vector, KMax)

s = 2
c10 = cp.Compressor(f"QSGD for fp64 s={s}")
c10.makeQSGD_FP64(levels = s, dInput = D)
checkCompressor(c10, base_vector, KMax)

print("END")

used_compressors = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]

compress_names = [c.name() + "\n(w=%g)" % (c.getW()) for c in used_compressors]
send_scalars = [c.really_need_to_send_components for c in used_compressors]

plt.barh(compress_names, [send/(KMax*D) for send in send_scalars], color=['lime', 'red', 'green', 'blue', 'cyan', 'royalblue', 'lightcoral', 'yellow', 'violet'])
plt.ylabel('Need to send components (fp64)')  
plt.title(f"Float component which are need to be send over the network for each compressor in average per single component. D={D}") 
# plt.tight_layout()
plt.show()

utils.printSystemInfo()

