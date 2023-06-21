#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdio>

constexpr double VERIFICATION_EPS = 1e-6;
constexpr int KERNEL_SIZE = 3;
constexpr double KERNEL[KERNEL_SIZE][KERNEL_SIZE] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

void gaussianBlur(double** image, double** blurredImage, int height, int width) {
    // Apply the Gaussian blur
    for (int j = 1; j < width - 1; ++j) {
        for (int i = 1; i < height - 1; ++i) {
            // Apply the kernel to the neighborhood
            double pixel = 0.0;
            for (int l = -1; l <= 1; ++l) {
                for (int k = -1; k <= 1; ++k) {
                    pixel += image[i + k][j + l] * KERNEL[k + 1][l + 1];
                }
            }

            // Update the blurred image
            blurredImage[i][j] = pixel / 16.0;
        }
    }
}

/*
    You can't change this function.
*/
double verificaion(double** image, int height, int width) {
    double sum = 0.0;
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            sum += image[i][j];
        }
    }
    return sum;
}

int main() {
    std::ifstream fileList("images.txt");

    std::vector<std::string> fileNames;
    std::string fileName;
    while (std::getline(fileList, fileName)) {
        fileNames.push_back(fileName);
    }

    double total_time = 0;
    for (const auto& inputFileName : fileNames) {
        // Open the input file for the current image
        std::ifstream inputFile(inputFileName);
        if (!inputFile.is_open()) {
            std::cerr << "Failed to open input file: " << inputFileName << std::endl;
            continue;
        } else {
            std::cout << "Opened input file: " << inputFileName << std::endl;
        }

        int height, width;

        // Read the image dimensions
        inputFile >> height >> width;

        // Dynamically allocate memory for the image and blurred image
        double** image = new double*[height];
        double** blurredImage = new double*[height];
        for (int i = 0; i < height; ++i) {
            image[i] = new double[width];
            blurredImage[i] = new double[width];
        }

        // Read the image pixels from the input file
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                inputFile >> image[i][j];
            }
        }

        // Read the verificaion from the input file
        double verification_sum;
        inputFile >> verification_sum;

        // Perform Gaussian blur
        auto start = std::chrono::steady_clock::now();
        gaussianBlur(image, blurredImage, height, width);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Write the blurred image pixels to the output file
        std::cout << "Execution time: " << duration.count() << " ms\n";
        total_time += duration.count() / 1000.0;

        // Verify the blurred image
        bool correct = true;
        double sum = verificaion(blurredImage, height, width);
        std::cout << sum << std::endl;
        if (fabs(sum - verification_sum) > VERIFICATION_EPS) {
            std::cout << "Verification: FAILED\n";
        }

        // Clean up memory
        for (int i = 0; i < height; ++i) {
            delete[] image[i];
            delete[] blurredImage[i];
        }
        delete[] image;
        delete[] blurredImage;

        // Close the input file
        inputFile.close();
    }
    std::printf("Total time: %.10f s\n", total_time);
    
    return 0;
}
