//Este bloque de código fue mi intento de hacerlo con base al código que nos mandó,
// no lo logré hacer funcionar

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define ARRAY_SIZE 64
#define BANK_SIZE 8

__global__ void sumColumns(int* array, int* result){
//no puedo sacar un array de dos dimensiones con el padArray, ni logro sacar los datos ya con padding
//Asi que solo podre sumar los datos sin padding

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < ARRAY_SIZE/BANK_SIZE) {
        int sum = 0;
        for (int i = 0; i <ARRAY_SIZE/BANK_SIZE; ++i) {
            int idx = i * (ARRAY_SIZE/BANK_SIZE) + tid;
            sum += array[idx];
        }

        result[tid] = sum;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void padArray(int* array, int* paddingResult) {
// Shared memory with padding
    __shared__ int sharedArray[ARRAY_SIZE + ARRAY_SIZE / BANK_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = bid * blockDim.x + tid;

// Load data into shared memory with padding
    sharedArray[tid] = array[index];
    __syncthreads();

// Access all keys from the original bank 0 in one clock pulse
    int offset = tid / BANK_SIZE;
    int newIndex = tid + offset;

// Use the modified index for accessing the padded shared memory
    int result = sharedArray[newIndex];
    paddingResult[index] = sharedArray[tid];
// Print the result for demonstration
    //printf("Thread %d: Original Value: %d, Padded Value: %d\n", tid, array[index], result);
}

int main() {
    srand(time(NULL));
    int array[ARRAY_SIZE];

// Initialize array values (you can replace this with your data)
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        if(i%8==0)
            printf("\n");
        array[i] = rand() % 10 + 1;
        printf("%i ", array[i]);
    }

    int* d_array;
    int* d_paddingResult;

// Allocate device memory
    cudaMalloc((void**)&d_array, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void**)&d_paddingResult, ARRAY_SIZE + ARRAY_SIZE / BANK_SIZE * sizeof(int));
    cudaMalloc((void**)&d_array, ARRAY_SIZE * sizeof(int));

// Copy array from host to device
    cudaMemcpy(d_array, array, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

// Define block and grid dimensions
    dim3 blockDim(BANK_SIZE);
    dim3 gridDim((ARRAY_SIZE + blockDim.x - 1) / blockDim.x);

// Launch kernel
    padArray<<<gridDim, blockDim>>>(d_array, d_paddingResult);

// Synchronize device to ensure print statements are displayed
    cudaDeviceSynchronize();


    int* paddingArray = (int *)malloc(ARRAY_SIZE + ARRAY_SIZE / BANK_SIZE * sizeof(int));
// Copy array from device to Host
    cudaMemcpy(paddingArray, d_paddingResult, ARRAY_SIZE + ARRAY_SIZE / BANK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    int* d_Result;
    int* h_Result = (int *)malloc(ARRAY_SIZE/BANK_SIZE  * sizeof(int));;
    cudaMalloc((void**)&d_Result, ARRAY_SIZE/BANK_SIZE * sizeof(int));

    sumColumns<<<gridDim, blockDim>>>(d_array, d_Result);

    cudaMemcpy(h_Result, d_Result, ARRAY_SIZE/BANK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n Sumas: \n");
    for (int i = 0; i < ARRAY_SIZE/BANK_SIZE; ++i) {
        printf("%i ", h_Result[i]);
    }
// Free allocated memory
    cudaFree(d_array);
    cudaFree(d_paddingResult);
    cudaFree(d_Result);
    free(h_Result);
    return 0;
}

//Este código lo hice con ayuda de ChatGPT, ta chido
/*
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 256

// Kernel para realizar la suma de los valores de cada columna de la matriz
__global__ void sumColumns(int *matrix, int width, int height, int padding, int *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < width + padding) {
        float sum = 0.0f;
        for (int i = 0; i < height; ++i) {
            int idx = i * (width + padding) + tid;
            sum += matrix[idx];
        }

        result[tid] = sum;
        tid += blockDim.x * gridDim.x;
    }
}

int main() {
    srand(time(NULL));

    int width = 5; // Tamaño de la matriz
    int height = 3;
    int padding = 2; // Tamaño del relleno

    // Dimensiones de la matriz con relleno
    int paddedWidth = width + padding;
    int paddedSize = paddedWidth * height;

    // Alojar memoria en el host para la matriz
    int *h_matrix = (int *)malloc(paddedSize * sizeof(int));

    // Inicializar la matriz con valores aleatorios
    for (int i = 0; i < paddedSize; ++i) {
        if(i%7==0)
            printf("\n");
        h_matrix[i] = rand() %10+1;
        printf("%i ", h_matrix[i]);
    }

    // Alojar memoria en el dispositivo para la matriz y el resultado
    int *d_matrix, *d_result;
    cudaMalloc((void **)&d_matrix, paddedSize * sizeof(int));
    cudaMalloc((void **)&d_result, paddedWidth * sizeof(int));

    // Transferir la matriz desde el host al dispositivo
    cudaMemcpy(d_matrix, h_matrix, paddedSize * sizeof(int), cudaMemcpyHostToDevice);

    // Calcular el número de bloques y hilos
    int numBlocks = (paddedWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(numBlocks, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    // Llamar al kernel para sumar las columnas de manera paralela
    sumColumns<<<grid, block>>>(d_matrix, width, height, padding, d_result);

    // Transferir el resultado desde el dispositivo al host
    int *h_result = (int *)malloc(paddedWidth * sizeof(int));
    cudaMemcpy(h_result, d_result, paddedWidth * sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimir el resultado
    printf("\n\nSuma de columnas:\n");
    for (int i = 0; i < paddedWidth; ++i) {
        printf("%i ", h_result[i]);
    }
    printf("\n");

    // Liberar la memoria
    free(h_matrix);
    free(h_result);
    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}*/