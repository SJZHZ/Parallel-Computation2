from PIL import Image
import numpy as np

def gaussian_blur(image):
    # Create a copy of the image
    blurred_image = np.zeros((image.shape[0] - 2, image.shape[1] - 2))

    # Define the Gaussian kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16.0

    # Get the height and width of the image
    height, width = image.shape[:2]

    # Apply the Gaussian blur
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Apply the kernel to the neighborhood
            pixel = 0.0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    pixel += image[i + k, j + l] * kernel[k + 1, l + 1]

            # Update the blurred image
            blurred_image[i - 1, j - 1] = pixel

    return blurred_image

def generate_verification(a):
    # Return sum of a
    return np.sum(a)

def image_to_text(image_path, output_path):
    # Open the image in grayscale mode
    image = Image.open(image_path).convert('L')
    width, height = image.size
    print width, height
    a = np.zeros((width, height))

    with open(output_path, 'w') as file:
        file.write('%d %d\n' % (height, width))
        for y in range(height):
            for x in range(width):
                # Get the pixel value at the current coordinates
                pixel_value = image.getpixel((x, y))
                a[x, y] = pixel_value
                
                # Write the pixel value to the file with 10 decimal places
                file.write('%d ' % pixel_value)
            file.write('\n')
        b = gaussian_blur(a)
        file.write('%.10f' % generate_verification(b))

    print "Image data saved to", output_path
    
image_to_text('images/potm2209a.jpg', 'images/potm2209a.txt')
image_to_text('images/potm2209b.jpg', 'images/potm2209b.txt')
image_to_text('images/potm2209c.jpg', 'images/potm2209c.txt')
image_to_text('images/potm2210a.jpg', 'images/potm2210a.txt')
image_to_text('images/potm2210c.jpg', 'images/potm2210c.txt')
image_to_text('images/potm2210d.jpg', 'images/potm2210d.txt')
image_to_text('images/potm2211a.jpg', 'images/potm2211a.txt')
image_to_text('images/potm2211b.jpg', 'images/potm2211b.txt')
image_to_text('images/potm2211c.jpg', 'images/potm2211c.txt')
image_to_text('images/potm2212a.jpg', 'images/potm2212a.txt')
