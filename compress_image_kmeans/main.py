import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.autoinit  # This automatically initializes the context


kernel_code = """
__global__ void assign_clusters(float *data, float *centroids, int *labels, int num_points, int num_centroids) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  // Global thread index

    if (idx < num_points) {
        float min_dist = 1e10;  // A large number representing infinity
        int closest_centroid = 0;

        // Find the closest centroid for each point
        for (int i = 0; i < num_centroids; i++) {
            float dist = 0.0;
            for (int j = 0; j < 3; j++) {  // Assuming 3 channels (RGB)
                float diff = data[idx * 3 + j] - centroids[i * 3 + j];
                dist += diff * diff;  // Squared Euclidean distance
            }
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = i;
            }
        }
        labels[idx] = closest_centroid;  // Assign the closest centroid to the label
    }
}

__global__ void update_centroids(float *data, float *centroids, int *labels, int num_points, int num_centroids) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  // Global thread index

    if (idx < num_centroids) {
        float sum[3] = {0.0, 0.0, 0.0};  // Sum of points for each centroid
        int count = 0;

        // Calculate the new centroid position
        for (int i = 0; i < num_points; i++) {
            if (labels[i] == idx) {
                for (int j = 0; j < 3; j++) {
                    sum[j] += data[i * 3 + j];
                }
                count++;
            }
        }

        // Update centroid only if count > 0 to avoid division by zero
        if (count > 0) {
            for (int j = 0; j < 3; j++) {
                centroids[idx * 3 + j] = sum[j] / count;
            }
        }
    }
}
"""


# Compile the kernel code
mod = compiler.SourceModule(kernel_code)


def cuda_kmeans(pixels, num_classes, max_iters=100):
    num_points = pixels.shape[0]

    # Allocate memory on the GPU
    data_ptr = cuda.mem_alloc(pixels.nbytes)
    centroids_ptr = cuda.mem_alloc(
        num_classes * pixels.shape[1] * pixels.dtype.itemsize)
    labels_ptr = cuda.mem_alloc(num_points * np.int32().itemsize)

    # Copy pixel data to GPU
    cuda.memcpy_htod(data_ptr, pixels)

    # Randomly initialize centroids
    centroids = np.random.rand(num_classes, 3).astype(np.float32)
    cuda.memcpy_htod(centroids_ptr, centroids)

    for _ in range(max_iters):
        # Assign clusters
        assign_clusters = mod.get_function("assign_clusters")
        block_size = 256
        grid_size = (num_points + block_size - 1) // block_size
        assign_clusters(data_ptr, centroids_ptr, labels_ptr, np.int32(
            num_points), np.int32(num_classes), block=(block_size, 1, 1), grid=(grid_size, 1))

        # Update centroids
        update_centroids = mod.get_function("update_centroids")
        update_centroids(data_ptr, centroids_ptr, labels_ptr, np.int32(num_points), np.int32(num_classes),
                         block=(block_size, 1, 1),
                         grid=((num_classes + block_size - 1) // block_size, 1))

    # Copy results back to host
    labels = np.empty(num_points, dtype=np.int32)
    centroids = np.empty((num_classes, 3), dtype=np.float32)
    cuda.memcpy_dtoh(labels, labels_ptr)
    cuda.memcpy_dtoh(centroids, centroids_ptr)

    # Free GPU memory
    data_ptr.free()
    centroids_ptr.free()
    labels_ptr.free()

    return labels, centroids


original_image = imread('Lenna.png')
# Flatten the image to (num_pixels, 3)
pixels = original_image.reshape(-1, 3).astype(np.float32)

# Run K-Means on GPU
num_classes = 16
labels, centroids = cuda_kmeans(pixels, num_classes)

# Create the compressed image
compressed_image = centroids[labels].reshape(original_image.shape)
print(f"compressed_image shape: {compressed_image.shape}")

# Save the compressed image
# plt.imsave('compressed_image_cuda.jpg', compressed_image.astype(np.uint8))
plt.imsave('compressed_image_cuda.png', compressed_image)
