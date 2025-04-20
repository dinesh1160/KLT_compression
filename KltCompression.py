import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

class KLTCompressor:
    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold

    def load_image(self):
        Tk().withdraw()  # hide the main window
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp"))]
        )
        if not file_path:
            raise Exception("No file selected.")
        image = cv2.imread(file_path)
        if image is None:
            raise Exception("Invalid image file.")
        return file_path, image


    def to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def compute_klt(self, image):
        mean_vector = np.mean(image, axis=0)
        centered = image - mean_vector
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx], mean_vector

    def select_components(self, eigenvalues):
        total = np.sum(eigenvalues)
        running_sum = 0
        for i, val in enumerate(eigenvalues):
            running_sum += val
            if running_sum / total >= self.variance_threshold:
                return i + 1
        return len(eigenvalues)

    def compress(self, image):
        eigenvalues, eigenvectors, mean_vector = self.compute_klt(image)
        num_components = self.select_components(eigenvalues)
        top_vectors = eigenvectors[:, :num_components]
        centered = image - mean_vector
        compressed = np.dot(centered, top_vectors)
        return compressed, top_vectors, mean_vector, num_components

    def reconstruct(self, compressed, eigenvectors, mean_vector):
        restored = np.dot(compressed, eigenvectors.T) + mean_vector
        return np.clip(restored, 0, 255).astype(np.uint8)

    def psnr(self, original, reconstructed):
        mse = np.mean((original - reconstructed) ** 2)
        return float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

    def visualize(self, original, reconstructed, num_components, file_path):
        psnr_value = self.psnr(original, reconstructed)
        print("Reconstructed with {} components, PSNR = {:.2f} dB".format(num_components, psnr_value))

        #save output
        import os
        base = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(os.path.dirname(file_path), base + "_klt.png")
        cv2.imwrite(output_path, reconstructed)
        print("Compressed image saved to:", output_path)

        # Show images
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(original, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title(f"KLT Reconstructed\n({num_components} components)")
        plt.imshow(reconstructed, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        plt.imshow(cv2.absdiff(original, reconstructed), cmap="hot")
        plt.axis("off")

        plt.suptitle(f"KLT Compression of {os.path.basename(file_path)}", fontsize=14)
        plt.tight_layout()
        plt.show()



def main():
    compressor = KLTCompressor(variance_threshold=0.95)
    try:
        file_path, image = compressor.load_image()
        gray = compressor.to_grayscale(image)
        compressed, eigvecs, mean_vec, n = compressor.compress(gray)
        restored = compressor.reconstruct(compressed, eigvecs, mean_vec)
        compressor.visualize(gray, restored, n, file_path)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
