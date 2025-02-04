# thrust_median
This sample showcases performing a vector addition in a traditional CUDA kernel, then passing the device pointer to thrust to perform some calculation. (CPU median code courtesy of Sam E. Whitebook)

This then does: data on CPU -> CUDA vector add kernel on GPU -> thrust median code on GPU -> return median value back to CPU.


Typically, on the first invocation, a ton of driver stuff happens, so the computations are fairly slow on the GPU the first time since thrust also handles much initialization. On subsequent runs, the speedups are noticeable.

Below are the results from some test systems:

```
Intel Xeon Gold 6148 @ 2.4GHz
2 threads per core
20 cores per socket
2 sockets

Tesla T4 GPU
```

Timings:
```
CPU Time to generate data: 56 ms 56705 us
CPU add time: 3 ms 3022 us
CPU Median: 1.000172
Time to find CPU median: 7 ms 7243 us
Thrust Time to generate data: 470 ms 470677 us
CUDA data transfer time (for generated data, coming from CPU generation): 1 ms 1786 us
CUDA vector add kernel time: 2 ms 2017 ms
Thrust Median 1.000163
Thrust Time to find median: 1 ms 1080 us
```

We can ignore the "thrust time to generate data", since that's just synthetic data generation for testing.

