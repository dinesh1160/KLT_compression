import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_klt(image):
    """
    Compute KLT basis vectors and eigenvalues from the image
    """
    # Flatten image into 2D array if needed
    rows, cols = image.shape
    mean_vector = np.mean(image, axis=0)
    centered_data = image - mean_vector

    # Compute covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    return eigenvectors, mean_vector

def klt_compress(image, num_components):
    """
    Compress image using KLT with top 'num_components' eigenvectors
    """
    eigenvectors, mean_vector = compute_klt(image)
    top_eigenvectors = eigenvectors[:, :num_components]

    centered_data = image - mean_vector
    compressed = np.dot(centered_data, top_eigenvectors)

    return compressed, top_eigenvectors, mean_vector

def klt_reconstruct(compressed_data, eigenvectors, mean_vector):
    """
    Reconstruct the image from compressed KLT representation
    """
    reconstructed = np.dot(compressed_data, eigenvectors.T) + mean_vector
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def main():
    # Load grayscale image
    image = cv2.imread("boat.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image not found.")
        return

    # Compress with KLT
    num_components = 50  # Adjust this value (e.g., 10, 50, 100, 200)
    compressed_data, eigvecs, mean_vec = klt_compress(image, num_components)

    # Reconstruct image
    reconstructed = klt_reconstruct(compressed_data, eigvecs, mean_vec)

    # Compute PSNR
    quality = psnr(image, reconstructed)
    print(f"PSNR with {num_components} components: {quality:.2f} dB")

    # Display results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"KLT Reconstructed\n({num_components} components)")
    plt.imshow(reconstructed, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(cv2.absdiff(image, reconstructed), cmap='hot')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
