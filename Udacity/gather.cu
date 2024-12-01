#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel: Perform gather operation
__global__ void gatherKernel(float *d_in, int *d_indices, float *d_out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Use the index array to fetch the corresponding element from input
        d_out[idx] = d_in[d_indices[idx]];
    }
}

int main()
{
    const int inputSize = 8;  // Size of the input array
    const int outputSize = 5; // Size of the output array
    const int bytesInput = inputSize * sizeof(float);
    const int bytesIndices = outputSize * sizeof(int);
    const int bytesOutput = outputSize * sizeof(float);

    // Host arrays
    float h_in[inputSize] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f}; // Input array
    int h_indices[outputSize] = {7, 4, 6, 0, 3};                                      // Indices to gather from
    float h_out[outputSize];                                                          // Output array (result)

    // Device arrays
    float *d_in, *d_out;
    int *d_indices;

    // Allocate device memory
    cudaMalloc((void **)&d_in, bytesInput);
    cudaMalloc((void **)&d_indices, bytesIndices);
    cudaMalloc((void **)&d_out, bytesOutput);

    // Copy data from host to device
    cudaMemcpy(d_in, h_in, bytesInput, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, bytesIndices, cudaMemcpyHostToDevice);

    // Launch kernel (1 block of 256 threads is enough for small sizes)
    int threadsPerBlock = 256;
    int blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
    gatherKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_indices, d_out, outputSize);

    // Copy results back to host
    cudaMemcpy(h_out, d_out, bytesOutput, cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "Input Array: ";
    for (int i = 0; i < inputSize; i++)
    {
        std::cout << h_in[i] << " ";
    }
    std::cout << "\nIndices Array: ";
    for (int i = 0; i < outputSize; i++)
    {
        std::cout << h_indices[i] << " ";
    }
    std::cout << "\nOutput Array: ";
    for (int i = 0; i < outputSize; i++)
    {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_indices);
    cudaFree(d_out);

    return 0;
}
