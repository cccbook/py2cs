from max.tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index

def main():
    height = 256
    width = 256
    channels = 3

    # Create the tensor of dimensions height, width, channels
    # and fill with random values.
    image = Tensor[DType.float32].rand(TensorShape(height, width, channels))

    # Declare the grayscale image.
    spec = TensorSpec(DType.float32, height, width)
    gray_scale_image = Tensor[DType.float32](spec)

    # Perform the RGB to grayscale transform.
    for y in range(height):
        for x in range(width):
            r = image[y, x, 0]
            g = image[y, x, 1]
            b = image[y, x, 2]
            gray_scale_image[Index(y, x)] = 0.299 * r + 0.587 * g + 0.114 * b

    print(gray_scale_image.shape())