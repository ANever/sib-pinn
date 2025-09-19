"""
********************************************************************************
device configuration
for details, see: https://www.tensorflow.org/guide/gpu
********************************************************************************
"""

import tensorflow as tf
from tensorflow.python.client import device_lib
import os
# flag = -1: use CPU
# flag =  0: use GPU:0
# flag =  1: use GPU:1


def config_gpu(flag=1, verbose=True):
    tf.debugging.set_log_device_placement(False)  # set True for careful device check
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    if flag == -1:
        if gpus:
            # no GPUs visible, use CPU only
            try:
                tf.config.experimental.set_visible_devices([], "GPU")
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                if verbose:
                    print(
                        len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
                    )
                    print("\nDevice information;")
                    print(device_lib.list_local_devices())
            except RuntimeError as e:
                print(e)
        

    elif flag == 0 or flag == 1:
        if gpus:
            # find GPU:0, with limited the memory growth
            try:
                tf.config.experimental.set_visible_devices(gpus[flag], "GPU")
                tf.config.experimental.set_memory_growth(gpus[flag], True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                if verbose:
                    print(
                        len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
                    )
                    print("\nDevice information;")
                    print(device_lib.list_local_devices())
            except RuntimeError as e:
                print(e)
        else:
            print("There are no available GPUs. Check hardware and CUDA Toolkit version.")

    else:
        raise NotImplementedError(">>>>> gpu_config: flag not specified properly")
