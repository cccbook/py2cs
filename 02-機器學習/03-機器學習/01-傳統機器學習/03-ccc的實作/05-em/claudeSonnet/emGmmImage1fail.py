import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage import io, color

def load_and_prepare_image(image_path):
    # Load image and convert to Lab color space
    image = io.imread(image_path)
    image_lab = color.rgb2lab(image)
    
    # Reshape the image to a 2D array of pixels
    pixels = image_lab.reshape(-1, 3)
    return image, pixels

def apply_gmm(pixels, n_components):
    # Initialize and fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(pixels)
    
    # Predict the labels for each pixel
    labels = gmm.predict(pixels)
    return labels, gmm

def create_segmented_image(image, labels, n_components):
    # Create a color map for the segments
    colors = plt.cm.rainbow(np.linspace(0, 1, n_components))
    
    # Create the segmented image
    segmented_image = colors[labels].reshape(image.shape)
    return segmented_image

def plot_results(original_image, segmented_image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(segmented_image)
    ax2.set_title('Segmented Image')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare the image
    image_path = "img/ccc.jpg"  # Replace with your image path
    image, pixels = load_and_prepare_image(image_path)
    
    # Apply GMM for image segmentation
    n_components = 5  # Number of segments
    labels, gmm = apply_gmm(pixels, n_components)
    
    # Create the segmented image
    segmented_image = create_segmented_image(image, labels, n_components)
    
    # Plot the results
    plot_results(image, segmented_image)
    
    # Print the means of each Gaussian component
    print("Means of Gaussian components:")
    for i, mean in enumerate(gmm.means_):
        print(f"Component {i + 1}: {mean}")

if __name__ == "__main__":
    main()