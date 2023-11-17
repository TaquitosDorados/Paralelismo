# Checking the Data Layout of Shared Memory

## Memoria Compartida Cuadrada

Se puede usar la memoria compartida al caché global con dimensiones cuadradas. 
Esto hace sencillo de calcular memorias desfasadas de una dimensión a partir de índices de hilos de dos dimensiones.

![Figura 5.12](./figura512.png)

Se puede declarar de manera estática de la siguiente manera:
`__shared__ int tile[N][N];`

Puedes acceder esta puedes usar bloques de hilos 2D:
```
    tile[threadIdx.y][threadIdx.x]
    tile[threadIdx.x][threadIdx.y]
```

Es mejor tener hilos con valores consecutivos de `threadIdx.x` accesando en ubicaciones consecutivas de la memoria compartida

### Accediendo a la Fila Mayor contra la Columna Mayor

Necesitamos un código que haga lo siguiente:

- Escribir indices de hilos globales a un arreglo de dos dimensiones en la memoria compartida en un orden de Fila Mayor
- Leer esos valores desde la memoria compartida en orden de Fila Mayor y guardarlos en la memoria global.

```
__global__ void setRowReadRow(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];
    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    // wait for all threads to complete
    __syncthreads();
    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}
```

Ya que los hilos en el mismo warp tienen valores consecutivos de `threadIdx.x` y usan este mismo para sacar el índice del dato dentro del arreglo de la memoria compartida, está libre de conflictos con el banco de memoria.

Por otro lado, si se cambian de lugar `threadIdx.y` y `threadIdx.x` cuando se asigne los datos al espacio del arreglo de la memoria compartida, la memoria accede al warp en un orden de Columna Mayor. Cada carga de la memoria compartida y su guardado causará un conflicto en los bancos de memoria.

```
__global__ void setColReadCol(int *out) {
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];
    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;
    // wait for all threads to complete
    __syncthreads();
    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}
```

### Escribiendo Fila Mayor y leyendo Columna Mayor

El siguiente código escribe a la memoria compartida en orden de Fila Mayor como en el anterior ejemplo.
Se implementa la asignación de valores a la memoria global desde la memoria compartida en un orden de Columna Mayor, cambiando de lugar los dos incides de los hilos al referenciar la memoria compartida: `out[idx] = tile[threadIdx.x][threadIdx.y];`

![Figura 5.13](./figura513.png)

```
__global__ void setRowReadCol(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];
    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;
    // wait for all threads to complete
    __syncthreads();
    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}
```

En este caso la operacion de guardado no presenta conflictos pero al cargarlo presenta 16.

### Memoria Compartida Dinámica

Se puede usar estos mismos kernels declarando una memoria compartida dinámica.
Esta tiene que ser declarada como un arreglo unidimensional sin tamaño.

Se puede escribir a la memoria compartida en un orden de fila mayor usando `row_idx` de manera calculada: `tile[row_idx] = row_idx;`

Usando la sincronización después de que el espacio en la memoria compartida haya sido llenado, puedes leerlo en un orden de columna mayor y asignar a la memoria global: `out[row_idx] = tile[col_idx];`

El código queda de la siguiente manera:

```
__global__ void setRowReadColDyn(int *out) {
    // dynamic shared memory
    extern __shared__ int tile[];
    // mapping from thread index to global memory index
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;
    // shared memory store operation
    tile[row_idx] = row_idx;
    // wait for all threads to complete
    __syncthreads();
    // shared memory load operation
    out[row_idx] = tile[col_idx];
}
```

El tamaño de la memoria compartida debe ser especificada cuando se lanza el kernel: `setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);` 

Este ejemplo sigue teniendo los 16 conflictos que en el código anterior.

//A partir de esta linea, me quede sin luz y tuve que trabajar desde la chamba xd

### Agregando Padding a la memoria compartida declarada de manera estática

Aplicando padding al arreglo de la memoria compartida es la unica forma de evitar conflictos con el banco de memoria.
Para este caso es sencillo, es simplemente agregar una columna extra al arreglo: `__shared__ int tile[BDIMY][BDIMX=1];`

En el codigo anterior, sabemos que existen 16 conflictos. Al aplicar padding a un elemento casa fila, los elementos de las columnas estan en diferentes bancos de memoria, eliminando estos conflictos.

### Agregando Padding a la memoria compartida declarada de manera dinamica

Aplicar padding en un arreglo dinamico es mas complicado. Se tiene que saltar un espacio por cada fila al aplicar la conversion de indices: 

```
    unsigned int row_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    unsigned int col_idx = threadIdx.x * (blockDim.x + 1) + threadIdx.y;
```

Ya que la memoria global que se usa para guardar datos en el kernel es mas pequenio que la memoria compartida que se le aplico padding, se necesitan tres indices: uno para la escritura de fila mayor a la memoria compartida, otro para lecturas de columna mayor desde la memoria mayor, y una ultima para accesos fucionados de la memoria global sin padding.

## Memoria Compartida Rectangular

Este se define como un arreglo en donde las filas y columnas no tienen el mismo tamanio.

No se puede simplememente cambiar de lugar las coordenadas de hilos al hacer una operacion de transposicion ya que viola el acceso a la memoria.

Funciona de la misma manera que la rectangular pero hay que poner atencion a los tamanios, dando indices independientes para cada columna y fila:

```
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;
```
