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
    float coordinates[2]; // x, y
};

struct pair_type {
    int key;
    Data point; //each data associate with a cluster (id == key)
};

struct KeyValueCompare {
    __host__ __device__ bool operator()(const pair_type& lhs, const pair_type& rhs) {
        return lhs.key < rhs.key;
    }
};

__device__ __host__
float squared_l2_distance(const Data& p1, const Data& p2) {
    float dist = 0;
    for (int i=0; i<2; i++) {
        int temp = p1.coordinates[i]-p2.coordinates[i];
        dist += temp * temp;
    }

    return dist;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__
void mapper(const Data* input, pair_type *pairs, Data *output, int k) {
    // Find centroid with min distance from the current point
    float min_distance = FLT_MAX;
    int cluster_id = -1;

    for (int i = 0; i < k; i++) {
        float dist = squared_l2_distance(*input, output[i]);
        if (dist < min_distance) {
            min_distance = dist;
            cluster_id = i;
        }
    }

   pairs->key = cluster_id;
   pairs->point = *input;
}

__global__
void mapKernel(const Data* input, pair_type *pairs, Data *dev_output, int k, int number_of_elements) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;    // Global id of the thread
    size_t jump = blockDim.x * gridDim.x; // Total number of threads, by jumping this much, it ensures that no thread gets the same data

    for (size_t i = threadId; i < number_of_elements; i += jump) {
        mapper(&input[i], &pairs[i * NUM_PAIRS], dev_output, k);
    }
}

void runMapper(const Data* dev_input, pair_type *dev_pairs, Data *dev_output, int blocks, int threads, int k, int number_of_elements) {
    mapKernel<<<blocks, threads>>>(dev_input, dev_pairs, dev_output, k, number_of_elements);
    cudaDeviceSynchronize();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ 
void reducer(pair_type *pairs, size_t len, Data *output,int k) {

    float new_values[2];    
    for (int i=0; i<2; i++)
        new_values[i] = 0;

    for (size_t i=0; i<len; i++) {
        for (int j=0; j<2; j++)
            new_values[j] += pairs[i].point.coordinates[j]; 
    }

    int cluster_idx = pairs[0].key;
    for (int i=0; i<2; i++) {
        new_values[i]/=len;
        output[cluster_idx].coordinates[i] = new_values[i];
    }

}


__global__ 
void reducerKernel(pair_type *pairs, Data *output,int k, int TOTAL_PAIRS) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;    
    size_t jump = blockDim.x * gridDim.x;

    for (size_t i=threadId; i<k; i+=jump) {
        
        size_t start_index = 0;                
        size_t end_index = TOTAL_PAIRS;      
        size_t uniq_key_index = 0;           
        size_t value_size = 0;          


        for (size_t j=1; j<TOTAL_PAIRS; j++) {
            if (KeyValueCompare()(pairs[j-1], pairs[j])) {
                if (uniq_key_index == i) {
                    end_index = j;
                    break; 
                }else {
                    uniq_key_index++;
                    start_index = j;
                }
            }
        }
        if (uniq_key_index != i) {return;}
        value_size = end_index - start_index;
        reducer(&pairs[start_index], value_size, &output[i],k);
    }
}


void runReducer(pair_type *dev_pairs, Data *dev_output, int blocks, int threads, int k, int TOTAL_PAIRS) {
    reducerKernel<<<blocks, threads>>>(dev_pairs, dev_output, k, TOTAL_PAIRS);
    cudaDeviceSynchronize();
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void runMapReduce(Data *input, Data *output,int k, int number_of_iterations,int number_of_elements, int blocks, int threads){
    Data *dev_input;
    Data *dev_output;
    pair_type *dev_pairs;

    size_t input_size = number_of_elements * sizeof(Data);
    cudaMalloc(&dev_input, input_size);

    const unsigned long long int TOTAL_PAIRS = number_of_elements * NUM_PAIRS;
    size_t pair_size = TOTAL_PAIRS * sizeof(pair_type);
    cudaMalloc(&dev_pairs, pair_size); // Allocate memory for key-value pairs

    size_t output_size = k * sizeof(Data);
    cudaMalloc(&dev_output, output_size);

    cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_output, output, output_size, cudaMemcpyHostToDevice);

    for (int iter=0; iter<number_of_iterations; iter++) {
        runMapper(dev_input,dev_pairs,dev_output, blocks, threads, k,number_of_elements);
        thrust::sort(thrust::device, dev_pairs, dev_pairs + TOTAL_PAIRS, KeyValueCompare());
        runReducer(dev_pairs, dev_output,blocks,threads,k, TOTAL_PAIRS );
    }
    cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_pairs);
    cudaFree(dev_output);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, const char* argv[]) {
    if (argc < 4) {
        std::cerr << "usage: k-means <data-file> <k> [iterations]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const auto k = std::atoi(argv[3]);//num of output/centroids
    const auto number_of_iterations = (argc == 5) ? std::atoi(argv[4]) : 300; 

    std::ifstream stream(argv[2]);
    std::string line;
    const size_t number_of_elements = std::count(std::istreambuf_iterator<char>(stream), 
                                      std::istreambuf_iterator<char>(), '\n'); //count lines
    stream.clear();
    stream.seekg(0);

    Data* input = new Data[number_of_elements];
    Data* input_copy = new Data[number_of_elements];
    Data* output = new Data[k];

    int x = 0;
    while (std::getline(stream, line)) {
        uint16_t label;
        std::istringstream line_stream(line);
        line_stream >> input[x].coordinates[0] >> input[x].coordinates[1] >> label;

        std::istringstream line_stream_copy(line);
        line_stream_copy >> input_copy[x].coordinates[0] >> input_copy[x].coordinates[1] >> label;
        x++;
    }

    std::mt19937 rng(std::random_device{}());
    std::shuffle(input_copy, input_copy + number_of_elements, rng);
    std::copy(input_copy, input_copy + k, output);

    const int threads = 1024;
    const int blocks = (number_of_elements + threads - 1) / threads;

    const auto start = std::chrono::high_resolution_clock::now();
    runMapReduce(input,output,k,number_of_iterations,number_of_elements,blocks,threads);//////////////
    const auto end = std::chrono::high_resolution_clock::now();
    

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    const auto duration =
      std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
      std::cerr << "MapReduce CUDA implementation Took: " << duration.count() << "s" << " for "<<number_of_elements<<" points."<<std::endl;
    
    std_time_used = duration.count();
    //cudaFree(d_counts);
    for (size_t cluster = 0; cluster < k; ++cluster) {
        //std::cout << output[cluster].coordinates[0] << " " << output[cluster].coordinates[1] << std::endl;
    }

    FILE *fp;
    int i;
    fp = fopen("Standardtimes.txt", "a");
        fprintf(fp, "%0.6f\n", std_time_used);
    fclose(fp);

    std::string str(std::to_string(number_of_elements)),str1,str2;
    //str = "output/standard/" + str;
    str2 = str + "_centroids.txt";

   fp = fopen(str2.c_str(), "w");
   for (i = 0; i < k; ++i) {
    fprintf(fp, "%0.6f %0.6f\n", output[i].coordinates[0], output[i].coordinates[1]);
    }
    fclose(fp);
  
}
