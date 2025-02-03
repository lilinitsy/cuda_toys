# thrust_median
This sample compares finding the median of a 1M element array on the CPU (CPU code courtesy of Sam E. Whitebrook) vs using Thrust.

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
CPU Time to generate data: 29 ms 29478 us
Time to find CPU median: 8 ms 8507 us

Thrust time to generate data: 26 ms 26970 us
Thrust Time to transfer data: 317ms 317541 us
Thrust time to find median: 3 ms  3020 us
```

Data transfer times are high, but if a large amount of data is batch transferred in a real use case, the improvement in speedups would likely be very noticeable; or if it were still using data that was already computed on the GPU, transfers would not have to occur.
