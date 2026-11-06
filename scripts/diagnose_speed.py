import os
import sys
import time

import tensorflow as tf

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if threads:
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

from src.inference import load_model_once

p = load_model_once()
size = p.img_size
x1 = tf.random.uniform((1, size, size, 3), 0, 255)
x8 = tf.random.uniform((8, size, size, 3), 0, 255)


def timeit(label, fn, n=5):
    fn()                                           # warm-up
    t = time.perf_counter()
    for _ in range(n):
        fn()
    print(f"{label}: {(time.perf_counter() - t) / n * 1000:.0f} ms per call")


print(f"CPU count: {os.cpu_count()} | threads: {threads or 'default'} | TF {tf.__version__}")
timeit("compiled call, 1 image ", lambda: p._forward(x1).numpy())
timeit("compiled call, 8 images", lambda: p._forward(x8).numpy())
timeit("model.predict, 1 image ", lambda: p.model.predict(x1, verbose=0))