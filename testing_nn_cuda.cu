#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

using namespace std;

// Testing image file name
const string testing_image_fn = "t10k-images.idx3-ubyte";

// Testing label file name
const string testing_label_fn = "t10k-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn = "testing-report.dat";

// Number of testing samples
const int nTesting = 10000;

// Image size in MNIST database
const int width = 28;
const int height = 28;

// n1 = Number of input neurons
// n2 = Number of hidden neurons
// n3 = Number of output neurons

const int n1 = width * height;  // = 784, without bias neuron
const int n2 = 128;
const int n3 = 10;  // Ten classes: 0 - 9

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *w1[n1 + 1], *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2[n2 + 1], *in2, *out2;

// Layer 3 - Output layer
double *in3, *out3;
double expected[n3 + 1];

// Image. In MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

// +--------------------+
// | About the software |
// +--------------------+

void about() {
    // Details
    cout << "*************************************************" << endl;
    cout << "*** Testing Neural Network for MNIST database ***" << endl;
    cout << "*************************************************" << endl;
    cout << endl;
    cout << "No. input neurons: " << n1 << endl;
    cout << "No. hidden neurons: " << n2 << endl;
    cout << "No. output neurons: " << n3 << endl;
    cout << endl;
    cout << "Testing image data: " << testing_image_fn << endl;
    cout << "Testing label data: " << testing_label_fn << endl;
    cout << "No. testing sample: " << nTesting << endl << endl;
}

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+

void init_array() {
    // Layer 1 - Layer 2 = Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        w1[i] = new double[n2 + 1];
    }

    out1 = new double[n1 + 1];

    // Layer 2 - Layer 3 = Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        w2[i] = new double[n3 + 1];
    }

    in2 = new double[n2 + 1];
    out2 = new double[n2 + 1];

    // Layer 3 - Output layer
    in3 = new double[n3 + 1];
    out3 = new double[n3 + 1];
}

// +----------------------------------------+
// | Load model of a trained Neural Network |
// +----------------------------------------+

void load_model(string file_name) {
    ifstream file(file_name.c_str(), ios::in);

    // Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            file >> w1[i][j];
        }
    }

    // Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            file >> w2[i][j];
        }
    }

    file.close();
}

// +------------------+
// | Sigmoid function |
// +------------------+

// double sigmoid(double x) {
//     return 1.0 / (1.0 + exp(-x));
// }

__global__ void compute_out1(double *in1, double **w1, double *out1, int n1,
                             int n2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n1) {
        out1[idx] = 0.0;
        for (int i = 0; i < n2; ++i) {
            out1[idx] += in1[i] * (*(w1[idx] + i));
        }
        out1[idx] = 1.0 / (1.0 + exp(-out1[idx]));
    }
}

__global__ void compute_in2(double *out1, double **w1, double *in2, int n1,
                            int n2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        in2[idx] = 0;
        for (int i = 1; i <= n1; ++i) {
            in2[idx] += out1[i - 1] * (*(w1[idx] + i - 1));
        }
    }
}
__global__ void compute_out2(double *in2, double **w2, double *out2, int n3,
                             int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n3) {
        out2[idx] = 0.0;
        for (int i = 0; i < n4; ++i) {
            out2[idx] += in2[i] * (*(w2[idx] + i));
        }
        out2[idx] = 1.0 / (1.0 + exp(-out2[idx]));
    }
}

__global__ void compute_in3(double *out2, double **w2, double *in3, int n2,
                            int n3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n3) {
        in3[idx] = 0;
        for (int i = 1; i <= n2; ++i) {
            in3[idx] += out2[i - 1] * (*(w2[idx] + i - 1));
        }
    }
}
__device__ double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

__global__ void sigmoid_kernel(double *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = sigmoid(arr[idx]);
    }
}

// double sigmoid(double x) {
//      return 1.0 / (1.0 + exp(-x));
//  }

void perceptron()

{
        std::cout << w1[0] << std::endl;
        std::cout << w1[1] << std::endl;
        double **w1_device, **w2_device;
    cudaMalloc((void **)&w1_device, (n1 + 1) * sizeof(double *));
    cudaMalloc((void **)&w2_device, (n2 + 1) * sizeof(double *));
    for (int i = 1; i <= n1; ++i) {
        double *row_device;
        cudaMalloc((void **)&row_device, (n2 + 1) * sizeof(double));
        cudaMemcpy(row_device, w1[i], (n2 + 1) * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(&w1_device[i], &row_device, sizeof(double *),
                   cudaMemcpyHostToDevice);
    }
    for (int i = 1; i <= n2; ++i) {
        double *row_device;
        cudaMalloc((void **)&row_device, (n3 + 1) * sizeof(double));
        cudaMemcpy(row_device, w2[i], (n3 + 1) * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(&w2_device[i], &row_device, sizeof(double *),
                   cudaMemcpyHostToDevice);
    }
         double *w1_0_host;
    cudaMemcpy(&w1_0_host, &w1_device[0], sizeof(double *),
               cudaMemcpyDeviceToHost);
    std::cout << (w1_0_host == nullptr ? "NULL" : "Not NULL") << std::endl;

    double *in1_d, *out1_d;
    cudaMalloc((void **)&in1_d, n1 * sizeof(double));
    cudaMalloc((void **)&out1_d, n1 * sizeof(double));

    double *d_in2;
    double *d_out2;
    cudaMalloc(&d_in2, n2 * sizeof(double));
    cudaMalloc(&d_out2, n2 * sizeof(double));

    double *d_in3;
    double *d_out3;
    cudaMalloc(&d_in3, n3 * sizeof(double));
    cudaMalloc(&d_out3, n3 * sizeof(double));

    cudaMemcpy(in1_d, out1, n1 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, in2, n2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in3, in3, n3 * sizeof(double), cudaMemcpyHostToDevice);

    // n1:

    dim3 block_size1(16, 16, 1);
    dim3 grid_size1((n1 + block_size1.x - 1) / block_size1.x,
                    (n2 + block_size1.y - 1) / block_size1.y, 1);
    compute_out1<<<grid_size1, block_size1>>>(in1_d, w1_device, out1_d, n1, n2);

    // define block size and grid size for compute_in2 and compute_out2
    int block_size_x = 16;
    int block_size_y = 16;
    int num_blocks_x = (n1 + block_size_x - 1) / block_size_x;
    int num_blocks_y = (n2 + block_size_y - 1) / block_size_y;
    dim3 grid_size(num_blocks_x, num_blocks_y, 1);
    dim3 block_size(block_size_x, block_size_y, 1);

    // call compute_in2 and compute_out2
    compute_in2<<<grid_size1, block_size>>>(out1, w1_device, d_in2, n2, n1);
    compute_out2<<<grid_size, block_size>>>(d_in2, w2_device, d_out2, n3, 128);
    sigmoid_kernel<<<grid_size, block_size>>>(in2, n2);

    // synchronize device and copy data back to host
    cudaDeviceSynchronize();
    cudaMemcpy(out2, d_out2, n2 * sizeof(double), cudaMemcpyDeviceToHost);

    // define block size and grid size for compute_in3
    int block_size_x3 = 16;
    int block_size_y3 = 16;
    int num_blocks_x3 = (n2 + block_size_x3 - 1) / block_size_x3;
    int num_blocks_y3 = (n3 + block_size_y3 - 1) / block_size_y3;
    dim3 grid_size3(num_blocks_x3, num_blocks_y3, 1);
    dim3 block_size3(block_size_x3, block_size_y3, 1);

    // call compute_in3 and sigmoid_kernel
    compute_in3<<<grid_size3, block_size3>>>(d_out2, w2_device, d_in3, n2, n3);
    sigmoid_kernel<<<grid_size3, block_size3>>>(in3, n3);

    // synchronize device and copy data back to host
    cudaDeviceSynchronize();
    cudaMemcpy(out3, d_out3, n3 * sizeof(double), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_in2);
    cudaFree(d_out2);
    cudaFree(d_in3);
    cudaFree(d_out3);
    cudaFree(w1_device);
    cudaFree(w2_device);
}

// +---------------+
// | Norm L2 error |
// +---------------+

double square_error() {
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
    }
    res *= 0.5;
    return res;
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+

int input() {
    // Reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
                d[i][j] = 0;
            } else {
                d[i][j] = 1;
            }
        }
    }

    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
        }
    }

    // Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i) {
        expected[i] = 0.0;
    }
    expected[number + 1] = 1.0;

    return (int)(number);
}

int main(int argc, char *argv[]) {
    about();

    report.open(report_fn.c_str(), ios::out);
    image.open(testing_image_fn.c_str(),
               ios::in | ios::binary);  // Binary image file
    label.open(testing_label_fn.c_str(),
               ios::in | ios::binary);  // Binary label file

    // Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
    }

    // Neural Network Initialization
    init_array();  // Memory allocation
    load_model(
        model_fn);  // Load model (weight matrices) of a trained Neural Network

    int nCorrect = 0;
    for (int sample = 1; sample <= nTesting; ++sample) {
        cout << "Sample " << sample << endl;

        // Getting (image, label)
        int label = input();

        // Classification - Perceptron procedure
        // perceptron(w1_device, w2_device);
        perceptron();

        // Prediction
        int predict = 1;
        for (int i = 2; i <= n3; ++i) {
            if (out3[i] > out3[predict]) {
                predict = i;
            }
        }
        --predict;

        // Write down the classification result and the squared error
        double error = square_error();
        printf("Error: %0.6lf\n", error);

        if (label == predict) {
            ++nCorrect;
            cout << "Classification: YES. Label = " << label
                 << ". Predict = " << predict << endl
                 << endl;
            report << "Sample " << sample << ": YES. Label = " << label
                   << ". Predict = " << predict << ". Error = " << error
                   << endl;
        } else {
            cout << "Classification: NO.  Label = " << label
                 << ". Predict = " << predict << endl;
            cout << "Image:" << endl;
            for (int j = 1; j <= height; ++j) {
                for (int i = 1; i <= width; ++i) {
                    cout << d[i][j];
                }
                cout << endl;
            }
            cout << endl;
            report << "Sample " << sample << ": NO.  Label = " << label
                   << ". Predict = " << predict << ". Error = " << error
                   << endl;
        }
    }

    // Summary
    printf("%s", "batee55");
    double accuracy = (double)(nCorrect) / nTesting * 100.0;
    cout << "Number of correct samples: " << nCorrect << " / " << nTesting
         << endl;
    printf("Accuracy: %0.2lf\n", accuracy);

    report << "Number of correct samples: " << nCorrect << " / " << nTesting
           << endl;
    report << "00 Accuracy: " << accuracy << endl;

    report.close();
    image.close();
    label.close();

    return 0;
}