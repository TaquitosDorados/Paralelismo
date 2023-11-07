# Examen Programacion Paralela 2

## SimpleMathAOS

```
==1011== NVPROF is profiling process 1011, command: ./simpleMathAoS
==1011== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1011== Profiling application: ./simpleMathAoS
==1011== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   80.19%  23.304ms         2  11.652ms  8.6804ms  14.623ms  [CUDA memcpy DtoH]
                   18.05%  5.2457ms         1  5.2457ms  5.2457ms  5.2457ms  [CUDA memcpy HtoD]
                    0.88%  256.10us         1  256.10us  256.10us  256.10us  warmup(innerStruct*, innerStruct*, int)
                    0.88%  255.85us         1  255.85us  255.85us  255.85us  testInnerStruct(innerStruct*, innerStruct*, int)
      API calls:   88.83%  566.38ms         2  283.19ms  376.90us  566.01ms  cudaMalloc
                    5.61%  35.742ms         1  35.742ms  35.742ms  35.742ms  cudaDeviceReset
                    4.90%  31.267ms         3  10.422ms  6.6946ms  15.576ms  cudaMemcpy
                    0.34%  2.1562ms         1  2.1562ms  2.1562ms  2.1562ms  cuDeviceGetPCIBusId
                    0.19%  1.2200ms         2  610.00us  446.90us  773.10us  cudaFree
                    0.11%  669.90us         2  334.95us  334.90us  335.00us  cudaDeviceSynchronize
                    0.03%  161.60us         2  80.800us  63.500us  98.100us  cudaLaunchKernel
                    0.00%  14.600us       101     144ns     100ns  1.3000us  cuDeviceGetAttribute
                    0.00%  6.1000us         1  6.1000us  6.1000us  6.1000us  cudaSetDevice
                    0.00%  5.8000us         2  2.9000us  2.6000us  3.2000us  cudaGetLastError
                    0.00%  4.7000us         1  4.7000us  4.7000us  4.7000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

La tarea que más consume tiempo dentro de los procesos de la GPU fue en `[CUDA memcpy DtoH]`, es decir que tardó un total de 23.304ms (80% del tiempo total de ejecución en la GPU) copiando datos de la GPU al Host, que en este caso es el procesador.

Después, el proceso inverso fue el que consumió casi la mayoría del tiempo restante (5.2ms), esto copiando los datos del Host hacia el Device, que es la GPU.

Ambas funciones de `warmup(innerStruct*, innerStruct*, int)` y `testInnerStruct(innerStruct*, innerStruct*, int)` duraron una cantidad insignificante de tiempo en comparación con lo anterior, en resumen podemos concluir que los intercambios de datos son los que más consumen tiempo en la GPU

En el caso de los procesos del procesador, la asignación de memoria fue la que tomó la mayoría del tiempo, un 88% (566.38 ms), de ahí los procesos a destacar son `cudaDeviceReset` (Limpiar los procesos de la GPU) con 35.7 ms y `cudaMemcpy` (Intercambio de datos con la GPU) con 31.2ms; de ahí los demás procesos van decayendo en tiempo significante del proceso total.

## SimpleMathSOA

```
==1027== NVPROF is profiling process 1027, command: ./simpleMathSoA
==1027== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1027== Profiling application: ./simpleMathSoA
==1027== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   73.35%  12.215ms         2  6.1076ms  3.7599ms  8.4554ms  [CUDA memcpy DtoH]
                   23.58%  3.9265ms         1  3.9265ms  3.9265ms  3.9265ms  [CUDA memcpy HtoD]
                    1.54%  256.42us         1  256.42us  256.42us  256.42us  warmup2(InnerArray*, InnerArray*, int)
                    1.54%  256.03us         1  256.03us  256.03us  256.03us  testInnerArray(InnerArray*, InnerArray*, int)
      API calls:   90.98%  584.89ms         2  292.45ms  380.00us  584.51ms  cudaMalloc
                    5.47%  35.165ms         1  35.165ms  35.165ms  35.165ms  cudaDeviceReset
                    2.89%  18.564ms         3  6.1881ms  3.9129ms  9.2690ms  cudaMemcpy
                    0.39%  2.4897ms         1  2.4897ms  2.4897ms  2.4897ms  cuDeviceGetPCIBusId
                    0.15%  981.80us         2  490.90us  359.80us  622.00us  cudaFree
                    0.11%  682.20us         2  341.10us  302.90us  379.30us  cudaDeviceSynchronize
                    0.01%  94.200us         2  47.100us  43.700us  50.500us  cudaLaunchKernel
                    0.00%  16.500us       101     163ns     100ns  1.4000us  cuDeviceGetAttribute
                    0.00%  5.9000us         1  5.9000us  5.9000us  5.9000us  cudaSetDevice
                    0.00%  5.1000us         1  5.1000us  5.1000us  5.1000us  cudaGetDeviceProperties
                    0.00%  4.7000us         2  2.3500us  2.3000us  2.4000us  cudaGetLastError
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%  1.2000us         1  1.2000us  1.2000us  1.2000us  cuDeviceGetName
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

Este ejemplo es muy similar al anterior, solo que en lugar de usar Arrays de Estructuras, utiliza Estructuras de Arrays.

El impacto en ambos dispositivos tienen una significativa diferencia con el ejemplo anterior. Parece ser que en este caso ha sido un proceso más rápido. En la GPU llegó a tener un proceso de transferencia de datos con el procesador con un tiempo recortado por casi la mitad, por ejemplo, `[CUDA memcpy DtoH]` con 12.215ms comparado con los 23.304ms del código anterior.

En el caso del procesador, no parece haber mucho cambio, exceptuando lo que podíamos imaginar con el `cudaMemcpy`, un tiempo más corto en transferir los datos ya que la GPU tardó alrededor de la mitad en terminar su trabajo.

## sumArrayZerocpy

```
==1049== NVPROF is profiling process 1049, command: ./sumArrayZerocpy
==1049== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1049== Profiling application: ./sumArrayZerocpy
==1049== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.33%  3.5200us         1  3.5200us  3.5200us  3.5200us  sumArraysZeroCopy(float*, float*, float*, int)
                   22.73%  2.4000us         2  1.2000us  1.1840us  1.2160us  [CUDA memcpy DtoH]
                   22.12%  2.3360us         1  2.3360us  2.3360us  2.3360us  sumArrays(float*, float*, float*, int)
                   21.82%  2.3040us         2  1.1520us     864ns  1.4400us  [CUDA memcpy HtoD]
      API calls:   94.24%  583.14ms         3  194.38ms  1.8000us  583.14ms  cudaMalloc
                    5.09%  31.475ms         1  31.475ms  31.475ms  31.475ms  cudaDeviceReset
                    0.35%  2.1756ms         1  2.1756ms  2.1756ms  2.1756ms  cuDeviceGetPCIBusId
                    0.16%  988.60us         2  494.30us  3.8000us  984.80us  cudaHostAlloc
                    0.06%  368.90us         2  184.45us  4.5000us  364.40us  cudaFreeHost
                    0.06%  358.00us         4  89.500us  33.100us  129.40us  cudaMemcpy
                    0.04%  218.20us         3  72.733us  2.5000us  208.10us  cudaFree
                    0.01%  60.300us         2  30.150us  28.600us  31.700us  cudaLaunchKernel
                    0.00%  14.900us       101     147ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  6.2000us         1  6.2000us  6.2000us  6.2000us  cudaSetDevice
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cudaGetDeviceProperties
                    0.00%  2.1000us         2  1.0500us     600ns  1.5000us  cudaHostGetDevicePointer
                    0.00%  1.6000us         3     533ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

Este ejemplo utiliza un método diferente para transferir datos entre la GPU y el Procesador, utilizando el `cudaMemcpy` solo para mandar una direccion y referencia del host al device, asi transfiriendo los datos mediante el uso del PCIe bus

Lo primero a notar es que los procesos dentro de la GPU tomaron un tiempo extremadamente corto, en el rango de los microsegundos. El proceso más tardado, aunque no por mucho, fue el esperado `sumArraysZeroCopy(float*, float*, float*, int)`, donde hace todos los calculos del vector. Los demás procesos del memcpy y la funcion de `sumArrays(float*, float*, float*, int)` tuvieron un tiempo de ejecución esencialmente iguales, dando así un completo balance entre los procesos de la GPU.

Como era de esperarse, en el procesador la tarea más pesada fue la de la asignación de memoria, de ahí los procesos no son significantes, nisiquiera los `cudaMemcpy`, que en los demás códigos por lo general es de los más tardados, ni `cuDeviceGetPCIBusId`, que es quien devuelve los valores de lo que fue procesado en la GPU.

## sumMatrixGPUManaged

```
==1071== NVPROF is profiling process 1071, command: ./sumMatrixGPUManaged
==1071== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1071== Profiling application: ./sumMatrixGPUManaged
==1071== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  12.948ms         2  6.4741ms  288.67us  12.660ms  sumMatrixGPU(float*, float*, float*, int, int)
      API calls:   91.39%  815.38ms         4  203.85ms  27.532ms  731.17ms  cudaMallocManaged
                    3.45%  30.801ms         1  30.801ms  30.801ms  30.801ms  cudaDeviceReset
                    3.31%  29.569ms         4  7.3922ms  7.2484ms  7.4490ms  cudaFree
                    1.52%  13.583ms         1  13.583ms  13.583ms  13.583ms  cudaDeviceSynchronize
                    0.24%  2.1681ms         1  2.1681ms  2.1681ms  2.1681ms  cuDeviceGetPCIBusId
                    0.07%  644.20us         2  322.10us  11.200us  633.00us  cudaLaunchKernel
                    0.00%  14.100us       101     139ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%  5.8000us         1  5.8000us  5.8000us  5.8000us  cudaSetDevice
                    0.00%  4.0000us         1  4.0000us  4.0000us  4.0000us  cudaGetDeviceProperties
                    0.00%  1.2000us         3     400ns     100ns     900ns  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cudaGetLastError
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

En este código se tiene un control total de las direcciones de memoria con las que se va a trabajar, indicandolo desde que se hace la asignación de memoria, esta forma se omite completamente el uso del Memcpy, ya que directamente los valores pueden ser modificados desde la GPU

El único proceso necesario dentro de la GPU es el de `sumMatrixGPU(float*, float*, float*, int, int)`, donde se hacen los calculos de la Matriz.

El proceso de asignación de memoria en el procesador fue varias veces más tardado que en el de los últimos códigos, con 815.38ms cuando antes rondaba entre los 200 ms, esto se esperaba ya que la asignación era más específica, pero sacrificando esto, hace que el proceso dentro de la GPU sea más sencilla y corta.

## sumMatrixGPUManual

```
==1089== NVPROF is profiling process 1089, command: ./sumMatrixGPUManual
==1089== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1089== Profiling application: ./sumMatrixGPUManual
==1089== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.52%  27.101ms         2  13.550ms  8.3698ms  18.731ms  [CUDA memcpy HtoD]
                   30.63%  12.669ms         1  12.669ms  12.669ms  12.669ms  [CUDA memcpy DtoH]
                    2.69%  1.1118ms         2  555.89us  288.73us  823.04us  sumMatrixGPU(float*, float*, float*, int, int)
                    1.16%  479.42us         2  239.71us  238.91us  240.51us  [CUDA memset]
      API calls:   87.57%  607.17ms         3  202.39ms  713.10us  605.72ms  cudaMalloc
                    6.50%  45.038ms         3  15.013ms  8.6183ms  23.545ms  cudaMemcpy
                    5.26%  36.474ms         1  36.474ms  36.474ms  36.474ms  cudaDeviceReset
                    0.33%  2.2576ms         1  2.2576ms  2.2576ms  2.2576ms  cuDeviceGetPCIBusId
                    0.19%  1.3256ms         3  441.87us  223.90us  799.30us  cudaFree
                    0.13%  929.30us         1  929.30us  929.30us  929.30us  cudaDeviceSynchronize
                    0.01%  62.700us         2  31.350us  24.300us  38.400us  cudaMemset
                    0.01%  62.500us         2  31.250us  28.200us  34.300us  cudaLaunchKernel
                    0.00%  15.600us       101     154ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  7.3000us         1  7.3000us  7.3000us  7.3000us  cudaSetDevice
                    0.00%  7.1000us         1  7.1000us  7.1000us  7.1000us  cudaGetDeviceProperties
                    0.00%  1.3000us         3     433ns     200ns     900ns  cuDeviceGetCount
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%  1.0000us         2     500ns     200ns     800ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cudaGetLastError
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

Este código es para comparar los tiempos en los procesos con el anterior código, donde se asignó la memoria de forma administrada, en este caso se usa con el Memcpy, procesando las mismas tareas.

Hay una enorme diferencia al ser comparada con los tiempos del código anterior. Como primer punto, el tiempo total de ejecución dentro de la GPU fue triplicada, esto es en su mayoria por la transferencia de datos. La función encargada de hacer las operaciones en realidad no tomó demasiado, tan solo 1ms.

En el caso del procesador, el tiempo de ejecución disminuyó considerablemente, pero a coste de un proceso en la GPU más lento. De esta forma podemos pensar en ambas formas de llevar a cabo estos procesos de transferencia de memoria, y elegir la que más nos convenga, procurando tener siempre un balance entre los procesos que hace la GPU y el procesador.

## transpose

```
==1111== NVPROF is profiling process 1111, command: ./transpose
==1111== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1111== Profiling application: ./transpose
==1111== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   86.82%  1.9853ms         1  1.9853ms  1.9853ms  1.9853ms  [CUDA memcpy HtoD]
                    6.62%  151.49us         1  151.49us  151.49us  151.49us  copyRow(float*, float*, int, int)
                    6.56%  150.02us         1  150.02us  150.02us  150.02us  warmup(float*, float*, int, int)
      API calls:   86.44%  634.10ms         2  317.05ms  434.00us  633.66ms  cudaMalloc
                   12.79%  93.791ms         1  93.791ms  93.791ms  93.791ms  cudaDeviceReset
                    0.32%  2.3634ms         1  2.3634ms  2.3634ms  2.3634ms  cudaMemcpy
                    0.31%  2.2569ms         1  2.2569ms  2.2569ms  2.2569ms  cuDeviceGetPCIBusId
                    0.07%  549.80us         2  274.90us  222.60us  327.20us  cudaFree
                    0.06%  404.50us         2  202.25us  166.80us  237.70us  cudaDeviceSynchronize
                    0.01%  57.000us         2  28.500us  15.400us  41.600us  cudaLaunchKernel
                    0.00%  16.500us       101     163ns     100ns  1.2000us  cuDeviceGetAttribute
                    0.00%  5.4000us         1  5.4000us  5.4000us  5.4000us  cudaSetDevice
                    0.00%  5.0000us         1  5.0000us  5.0000us  5.0000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     100ns  1.1000us  cuDeviceGetCount
                    0.00%  1.3000us         2     650ns     600ns     700ns  cudaGetLastError
                    0.00%  1.0000us         2     500ns     200ns     800ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

Este código solo es un ejemplo para la optimización del proceso de transponer una matriz usando CUDA

En este caso no hay realmente nada nuevo que ver. En la GPU tenemos lo que conocemos que por lo general será siempre lo más tardado: `[CUDA memcpy HtoD]`, la transferencia de datos, que no llevó más de 2ms. La función `copyRow(float*, float*, int, int)`, la cual hace los calculos, tuvo un tiempo mínimo, de tan solo 150 microsegundos.

En el procesador, la asignación de memoria y la transferencia de datos entre dispositivos fueron las más destacables.

## writeSegment

```
==1127== NVPROF is profiling process 1127, command: ./writeSegment
==1127== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==1127== Profiling application: ./writeSegment
==1127== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.98%  2.1129ms         3  704.29us  518.98us  921.45us  [CUDA memcpy DtoH]
                   29.36%  940.23us         2  470.12us  465.19us  475.04us  [CUDA memcpy HtoD]
                    1.55%  49.504us         1  49.504us  49.504us  49.504us  writeOffset(float*, float*, float*, int, int)
                    1.49%  47.712us         1  47.712us  47.712us  47.712us  warmup(float*, float*, float*, int, int)
                    0.91%  29.120us         1  29.120us  29.120us  29.120us  writeOffsetUnroll2(float*, float*, float*, int, int)
                    0.72%  23.072us         1  23.072us  23.072us  23.072us  writeOffsetUnroll4(float*, float*, float*, int, int)
      API calls:   92.61%  579.23ms         3  193.08ms  301.40us  578.59ms  cudaMalloc
                    6.01%  37.576ms         1  37.576ms  37.576ms  37.576ms  cudaDeviceReset
                    0.83%  5.1802ms         5  1.0360ms  537.40us  2.0100ms  cudaMemcpy
                    0.34%  2.1550ms         1  2.1550ms  2.1550ms  2.1550ms  cuDeviceGetPCIBusId
                    0.11%  687.50us         3  229.17us  186.80us  276.10us  cudaFree
                    0.06%  399.00us         4  99.750us  72.600us  145.30us  cudaDeviceSynchronize
                    0.04%  225.60us         4  56.400us  20.100us  89.100us  cudaLaunchKernel
                    0.00%  14.400us       101     142ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  4.1000us         1  4.1000us  4.1000us  4.1000us  cudaSetDevice
                    0.00%  3.3000us         4     825ns     400ns  1.1000us  cudaGetLastError
                    0.00%  1.2000us         3     400ns     200ns     800ns  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
```

Este código muestra el impacto que tiene las escrituras no alineadas, llamando varias veces al kernel, trasfiriendo y regresando datos una y otra vez.

Lo único que podemos notar es el aumento de veces que se tuvo que hacer la transferencia de datos, cosa que en el rendimiento impactó un poco en los tiempos de ejecución pero no fue la gran cosa. Supongo que mientras mas llames a hacer estas transferencias de memoria, más tardará.

## memTransfer

```
==935== NVPROF is profiling process 935, command: ./memTransfer
==935== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==935== Profiling application: ./memTransfer
==935== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.15%  2.1117ms         1  2.1117ms  2.1117ms  2.1117ms  [CUDA memcpy HtoD]
                   47.85%  1.9374ms         1  1.9374ms  1.9374ms  1.9374ms  [CUDA memcpy DtoH]
      API calls:   93.74%  577.35ms         1  577.35ms  577.35ms  577.35ms  cudaMalloc
                    5.15%  31.729ms         1  31.729ms  31.729ms  31.729ms  cudaDeviceReset
                    0.71%  4.3856ms         2  2.1928ms  2.1784ms  2.2072ms  cudaMemcpy
                    0.34%  2.0994ms         1  2.0994ms  2.0994ms  2.0994ms  cuDeviceGetPCIBusId
                    0.05%  306.30us         1  306.30us  306.30us  306.30us  cudaFree
                    0.00%  14.700us       101     145ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  8.0000us         1  8.0000us  8.0000us  8.0000us  cudaSetDevice
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

En este código el objetivo es transferir por medio del Memcpy la memoria del host y meterlo en un array dentro de la GPU, para que dentro de este acceda a esa memoria y pueda cambiar sus datos.

No hay mucho que comentar en este código, ya que realmente no se procesa nada más que la transferencia de memoria. No hay ninguna función que analizar, simplemente ver como CUDA hace su trabajo a gran velocidad

## pinMemTransfer

```
==947== NVPROF is profiling process 947, command: ./pinMemTransfer
==947== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==947== Profiling application: ./pinMemTransfer
==947== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.57%  1.3036ms         1  1.3036ms  1.3036ms  1.3036ms  [CUDA memcpy HtoD]
                   49.43%  1.2743ms         1  1.2743ms  1.2743ms  1.2743ms  [CUDA memcpy DtoH]
      API calls:   93.65%  564.84ms         1  564.84ms  564.84ms  564.84ms  cudaHostAlloc
                    5.15%  31.051ms         1  31.051ms  31.051ms  31.051ms  cudaDeviceReset
                    0.45%  2.7319ms         2  1.3660ms  1.3368ms  1.3951ms  cudaMemcpy
                    0.34%  2.0604ms         1  2.0604ms  2.0604ms  2.0604ms  cuDeviceGetPCIBusId
                    0.30%  1.8091ms         1  1.8091ms  1.8091ms  1.8091ms  cudaFreeHost
                    0.06%  342.90us         1  342.90us  342.90us  342.90us  cudaMalloc
                    0.04%  261.00us         1  261.00us  261.00us  261.00us  cudaFree
                    0.00%  15.400us       101     152ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%  7.2000us         1  7.2000us  7.2000us  7.2000us  cudaSetDevice
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cudaGetDeviceProperties
                    0.00%  1.0000us         3     333ns     100ns     700ns  cuDeviceGetCount
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

El punto de este código es exactamente el mismo que el anterior, con la diferencia de que la asignación de memoria está fijada en el host.

Comparado con la anterior, con este método se logró ahorrarse unos cuantos milisegundos tanto en la GPU como en el procesador, supongo que esto se debe a que no se necesita hacer la asignación de memoria pensando en la GPU tambien.

## readSegment

```
==963== NVPROF is profiling process 963, command: ./readSegment
==963== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==963== Profiling application: ./readSegment
==963== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.71%  992.10us         1  992.10us  992.10us  992.10us  [CUDA memcpy DtoH]
                   45.41%  906.47us         2  453.23us  447.23us  459.23us  [CUDA memcpy HtoD]
                    2.48%  49.408us         1  49.408us  49.408us  49.408us  readOffset(float*, float*, float*, int, int)
                    2.40%  48.001us         1  48.001us  48.001us  48.001us  warmup(float*, float*, float*, int, int)
      API calls:   93.88%  603.77ms         3  201.26ms  313.00us  603.14ms  cudaMalloc
                    5.02%  32.299ms         1  32.299ms  32.299ms  32.299ms  cudaDeviceReset
                    0.52%  3.3638ms         3  1.1213ms  585.30us  2.1168ms  cudaMemcpy
                    0.40%  2.5464ms         1  2.5464ms  2.5464ms  2.5464ms  cuDeviceGetPCIBusId
                    0.13%  833.20us         3  277.73us  167.00us  455.50us  cudaFree
                    0.03%  206.30us         2  103.15us  68.900us  137.40us  cudaDeviceSynchronize
                    0.01%  65.800us         2  32.900us  16.900us  48.900us  cudaLaunchKernel
                    0.00%  15.800us       101     156ns     100ns  1.4000us  cuDeviceGetAttribute
                    0.00%  5.5000us         1  5.5000us  5.5000us  5.5000us  cudaSetDevice
                    0.00%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  1.2000us         2     600ns     600ns     600ns  cudaGetLastError
                    0.00%     900ns         3     300ns     100ns     600ns  cuDeviceGetCount
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

Este código funciona de forma similar a la que era para hacer escrituras, en este caso es para lectura y su impacto en el rendimiento.

No hay mucho que explicar mas que el hecho de que estén mal alineadas hace que el proceso sea un poco más lento.

## readSegmentUnroll

```
==985== NVPROF is profiling process 985, command: ./readSegmentUnroll
==985== Warning: Unified Memory Profiling is not supported on the current configuration because a pair of devices without peer-to-peer support is detected on this multi-GPU setup. When peer mappings are not available, system falls back to using zero-copy memory. It can cause kernels, which access unified memory, to run slower. More details can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory
==985== Profiling application: ./readSegmentUnroll
==985== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.13%  2.0672ms         3  689.07us  470.56us  864.49us  [CUDA memcpy DtoH]
                   27.79%  895.65us         2  447.83us  446.53us  449.12us  [CUDA memcpy HtoD]
                    1.94%  62.593us         4  15.648us  15.360us  16.320us  [CUDA memset]
                    1.56%  50.368us         1  50.368us  50.368us  50.368us  readOffsetUnroll4(float*, float*, float*, int, int)
                    1.55%  49.984us         1  49.984us  49.984us  49.984us  readOffset(float*, float*, float*, int, int)
                    1.54%  49.632us         1  49.632us  49.632us  49.632us  readOffsetUnroll2(float*, float*, float*, int, int)
                    1.49%  47.904us         1  47.904us  47.904us  47.904us  warmup(float*, float*, float*, int, int)
      API calls:   93.30%  592.46ms         3  197.49ms  309.10us  591.77ms  cudaMalloc
                    5.46%  34.676ms         1  34.676ms  34.676ms  34.676ms  cudaDeviceReset
                    0.69%  4.4052ms         5  881.04us  498.20us  1.8633ms  cudaMemcpy
                    0.32%  2.0617ms         1  2.0617ms  2.0617ms  2.0617ms  cuDeviceGetPCIBusId
                    0.12%  749.60us         3  249.87us  170.00us  390.60us  cudaFree
                    0.06%  357.30us         4  89.325us  71.700us  130.70us  cudaDeviceSynchronize
                    0.02%  144.90us         4  36.225us  22.500us  52.700us  cudaMemset
                    0.01%  91.300us         4  22.825us  9.4000us  47.700us  cudaLaunchKernel
                    0.00%  14.600us       101     144ns     100ns  1.3000us  cuDeviceGetAttribute
                    0.00%  6.9000us         1  6.9000us  6.9000us  6.9000us  cudaGetDeviceProperties
                    0.00%  5.7000us         1  5.7000us  5.7000us  5.7000us  cudaSetDevice
                    0.00%  2.4000us         4     600ns     500ns     700ns  cudaGetLastError
                    0.00%  1.5000us         2     750ns     300ns  1.2000us  cuDeviceGet
                    0.00%  1.2000us         3     400ns     100ns     900ns  cuDeviceGetCount
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

Este código tiene la misma funcionalidad que la anterior, pero le agrega unos procesos de desenrrollamiento para hacer el proceso más optimo

A pesar de que el punto de este código sea un mejor rendimiento, viendo los resultados, vemos que hizo todo lo contrario. Aumentó enormemente el tiempo de ejecución dentro de la GPU. En el procesador no parece que haya bajado el tiempo de ejecución sumando todos los tiempos de cada proceso. Quizás este código funcione de manera más optima para procesos más grandes.

