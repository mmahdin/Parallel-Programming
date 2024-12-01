#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel: Apply the square operation (map function)
__global__ void squareKernel(float *d_in, float *d_out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate thread index
    if (idx < size)
    { // Ensure we don't go out of bounds
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

int main()
{
    const int arraySize = 10;                    // Number of elements in the array
    const int bytes = arraySize * sizeof(float); // Size in bytes

    // Host arrays
    float h_in[arraySize];  // Input array
    float h_out[arraySize]; // Output array

    // Initialize input array
    for (int i = 0; i < arraySize; i++)
    {
        h_in[i] = static_cast<float>(i + 1); // Fill with values 1, 2, ..., 10
    }

    // Device arrays
    float *d_in, *d_out;

    // Allocate device memory
    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);

    // Copy data from host to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    const int threadsPerBlock = 256;
    const int blocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks
    squareKernel<<<blocks, threadsPerBlock>>>(d_in, d_out, arraySize);

    // Copy data back from device to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "Input Array:  ";
    for (int i = 0; i < arraySize; i++)
    {
        std::cout << h_in[i] << " ";
    }
    std::cout << "\nOutput Array: ";
    for (int i = 0; i < arraySize; i++)
    {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
