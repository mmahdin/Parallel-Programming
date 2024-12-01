#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel: Perform scatter operation
__global__ void scatterKernel(float *d_in, int *d_indices, float *d_out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Place the input element at the position specified by the index array
        d_out[d_indices[idx]] = d_in[idx];
    }
}

int main()
{
    const int inputSize = 5;  // Size of the input array
    const int outputSize = 8; // Size of the output array (target array)
    const int bytesInput = inputSize * sizeof(float);
    const int bytesIndices = inputSize * sizeof(int);
    const int bytesOutput = outputSize * sizeof(float);

    // Host arrays
    float h_in[inputSize] = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f}; // Input array
    int h_indices[inputSize] = {3, 0, 7, 1, 5};                       // Indices to scatter into
    float h_out[outputSize] = {0.0f};                                 // Output array initialized to zero

    // Device arrays
    float *d_in, *d_out;
    int *d_indices;

    // Allocate device memory
    cudaMalloc((void **)&d_in, bytesInput);
    cudaMalloc((void **)&d_indices, bytesIndices);
    cudaMalloc((void **)&d_out, bytesOutput);

    // Initialize the device output array to zero
    cudaMemset(d_out, 0, bytesOutput);

    // Copy data from host to device
    cudaMemcpy(d_in, h_in, bytesInput, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, bytesIndices, cudaMemcpyHostToDevice);

    // Launch kernel (1 block of 256 threads is sufficient for small sizes)
    int threadsPerBlock = 256;
    int blocksPerGrid = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    scatterKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, h_indices, d_out, inputSize);

    // Copy results back to host
    cudaMemcpy(h_out, d_out, bytesOutput, cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "Input Array: ";
    for (int i = 0; i < inputSize; i++)
    {
        std::cout << h_in[i] << " ";
    }
    std::cout << "\nIndices Array: ";
    for (int i = 0; i < inputSize; i++)
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
