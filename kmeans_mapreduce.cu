#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

double std_time_used;
const int NUM_PAIRS = 1;

struct Data {
  Data(int size) : size(size), bytes(size * sizeof(float)) {
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMemset(x, 0, bytes);
    cudaMemset(y, 0, bytes);
  }

  Data(int size, std::vector<float>& h_x, std::vector<float>& h_y)
  : size(size), bytes(size * sizeof(float)) {
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMemcpy(x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_y.data(), bytes, cudaMemcpyHostToDevice);
  }

  ~Data() {
    cudaFree(x);
    cudaFree(y);
  }

  float* x{nullptr};
  float* y{nullptr};
  int size{0};
  int bytes{0};
};

struct pair_type {
    int key;
    Data value;
};//each data associate with a cluster (id == key)

struct KeyValueCompare {
    __host__ __device__ bool operator()(const pair_type& lhs, const pair_type& rhs) {
        return lhs.key < rhs.key;
    }
};

__device__ 
float
squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__device__
void mapper(const Data* input, pair_type *pairs, Data *output, int k) {
    // Find centroid with min distance from the current point
    float min_distance = FLT_MAX;
    int cluster_id = -1;

    for (int i = 0; i < k; i++) {
        float dist = squared_l2_distance(input->x[0], input->y[0], output[i].x[0], output[i].y[0]);
        if (dist < min_distance) {
            min_distance = dist;
            cluster_id = i;
        }
    }

    pairs->key = cluster_id;
    pairs->value = *input;
}


__global__
void mapKernel(const Data* input, pair_type *pairs, Data *dev_output, int k, int number_of_elements) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;    // Global id of the thread
    // Total number of threads, by jumping this much, it ensures that no thread gets the same data
    size_t jump = blockDim.x * gridDim.x;

    for (size_t i = threadId; i < number_of_elements; i += jump) {
        mapper(&input[i], &pairs[i * NUM_PAIRS], dev_output, k);
    }
}

void runMapper(const Data* dev_input, pair_type *dev_pairs, Data *dev_output, int blocks, int threads, int k,int number_of_elements) {
    mapKernel<<<blocks, threads>>>(dev_input, dev_pairs, dev_output, k,number_of_elements);
    cudaDeviceSynchronize();
}

__device__
void reducer(pair_type *pairs, size_t len, Data *output, int k) {
    // Calculate the sum of values for each dimension
    float sum_x = 0.0f;
    float sum_y = 0.0f;

    for (size_t i = 0; i < len; i++) {
        sum_x += pairs[i].value.x[0];
        sum_y += pairs[i].value.y[0];
    }

    // Calculate the average value for each dimension
    float avg_x = sum_x / len;
    float avg_y = sum_y / len;

    // Take the key of any pair
    int cluster_idx = pairs[0].key;

    // Update the output centroid coordinates
    cudaMemcpy(&(output[cluster_idx].x[0]), &avg_x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(output[cluster_idx].y[0]), &avg_y, sizeof(float), cudaMemcpyHostToDevice);
}


__global__
void reducerKernel(pair_type *pairs, Data *output, int k, int TOTAL_PAIRS) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;    // Global id of the thread
    // Total number of threads, by jumping this much, it ensures that no thread gets the same data
    size_t jump = blockDim.x * gridDim.x;

    for (size_t i=threadId; i<k; i+=jump) {
        size_t start_index = 0;                // Starting index of the key in the array of pairs
        size_t end_index = TOTAL_PAIRS;        // Ending index of the key in array of pairs
        size_t uniq_key_index = 0;                // In a list of unique sorted keys, the index of the key
        size_t value_size = 0;                 // No. of pairs for this key

        for (size_t j=1; j<TOTAL_PAIRS; j++) {
            if (KeyValueCompare()(pairs[j-1], pairs[j])) {
                // The keys are unequal, therefore we have moved on to a new key
                if (uniq_key_index == i) {
                    end_index = j;
                    break;
                }
                else {
                    uniq_key_index++;
                    start_index = j;
                }
            }
        }

        if (uniq_key_index != i) {
            return;             // Enjoy, nothing to be done!
        }
        value_size = end_index - start_index;

        reducer(&pairs[start_index], value_size, &output[i],k);
        }
}


void runReducer(pair_type *dev_pairs, Data *dev_output,int blocks, int threads, int k, int  TOTAL_PAIRS) {
    reducerKernel<<<blocks, threads>>>(dev_pairs, dev_output,k, TOTAL_PAIRS);
    cudaDeviceSynchronize();
}


void runMapReduce(const Data* input, Data *output,int k, int number_of_iterations,int number_of_elements, int blocks, int threads) {
    
    Data *dev_input;
    Data *dev_output;
    pair_type *dev_pairs;

    size_t input_size = number_of_elements * sizeof(Data);
    cudaMalloc(&dev_input, input_size);

    const unsigned long long int TOTAL_PAIRS = number_of_elements*number_of_elements;
    size_t pair_size = TOTAL_PAIRS * sizeof(Data);
    cudaMalloc(&dev_pairs, pair_size);

    size_t output_size = k * sizeof(Data);
    cudaMalloc(&dev_output, output_size);
    cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_output, output, output_size, cudaMemcpyHostToDevice);
    for (int iter = 0; iter < number_of_iterations; iter++) {

        runMapper(dev_input,dev_pairs,dev_output, blocks, threads, k,number_of_elements);
        thrust::sort(thrust::device, dev_pairs, dev_pairs + TOTAL_PAIRS, KeyValueCompare());
        runReducer(dev_pairs, dev_output,blocks,threads,k, TOTAL_PAIRS );
    }

    cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_input);
    cudaFree(dev_pairs);
    cudaFree(dev_output);
}

////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, const char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: k-means <data-file> <k> [iterations]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const auto k = std::atoi(argv[3]);
  const auto number_of_iterations = (argc == 5) ? std::atoi(argv[4]) : 300;

  std::vector<float> h_x;
  std::vector<float> h_y;
  std::ifstream stream(argv[2]);
  std::string line;
  while (std::getline(stream, line)) {
    std::istringstream line_stream(line);
    float x, y;
    uint16_t label;
    line_stream >> x >> y >> label;
    h_x.push_back(x);
    h_y.push_back(y);
  }

  const size_t number_of_elements = h_x.size();
  

  std::mt19937 rng(std::random_device{}());
  std::shuffle(h_x.begin(), h_x.end(), rng);
  std::shuffle(h_y.begin(), h_y.end(), rng);
 
  const int threads = 1024;
  const int blocks = (number_of_elements + threads - 1) / threads;

  int* d_counts;
  cudaMalloc(&d_counts, k * blocks * sizeof(int));
  cudaMemset(d_counts, 0, k * blocks * sizeof(int));
  
  const auto start = std::chrono::high_resolution_clock::now();

  //Map Reduce Here
  Data input(number_of_elements, h_x, h_y);
  Data output(k, h_x, h_y);
  runMapReduce(input, output, k, number_of_iterations, number_of_elements, blocks,threads);

  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
  std::cerr << "Standard CUDA implementation Took: " << duration.count() << "s" << " for "<<h_x.size()<<" points."<<std::endl;

  std_time_used = duration.count();

  cudaFree(d_counts);

  std::vector<float> mean_x(k, 0);
  std::vector<float> mean_y(k, 0);
  cudaMemcpy(mean_x.data(), output.x, output.bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(mean_y.data(), output.y, output.bytes, cudaMemcpyDeviceToHost);

  for (size_t cluster = 0; cluster < k; ++cluster) {
    //std::cout << mean_x[cluster] << " " << mean_y[cluster] << std::endl;
  }

  FILE *fp;
  int i;

  fp = fopen("Standardtimes.txt", "a");
    fprintf(fp, "%0.6f\n", std_time_used);
  fclose(fp);

  std::string str(std::to_string(h_x.size())),str1,str2;
  str = "output/standard/" + str;

  str2 = str + "_centroids.txt";
  fp = fopen(str2.c_str(), "w");
  for(i = 0; i < k; ++i){
    fprintf(fp, "%0.6f %0.6f\n", mean_x[i], mean_y[i]);
  }
  fclose(fp);


}