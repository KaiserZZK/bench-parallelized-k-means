#include <stdio.h>
#include <time.h>
#include <math.h>

#define N 64
#define TPB 1024

__device__ float distance(float x1, float x2, float y1, float y2) {
	return pow((y1-x1),2) + pow((y2-x2),2);
}

__global__ void kMeansClusterAssignment(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_k, int *d_n) {
	//get idx for this datapoint
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= d_n) return;

	//find the closest centroid to this datapoint
	float min_dist = INFINITY;
	int closest_centroid = 0;

	for(int c = 0; c<d_k;++c) { // note, now variables like k and n must be passed into the function to work
		float dist = distance(d_datapoints[2*idx], d_datapoints[2*idx+1], d_centroids[2*c], d_centroids[2*c+1]);

		if(dist < min_dist) {
			min_dist = dist;
			closest_centroid=c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx]=closest_centroid;
}


__global__ void kMeansCentroidUpdate(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clust_sizes, int *d_k, int *d_n) {
	//get idx of thread at grid level
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= d_n) return;

	//get idx of thread at the block level
	const int s_idx = threadIdx.x;

	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__shared__ float s_datapoints[2*TPB];
	s_datapoints[2*s_idx]= d_datapoints[2*idx];
    s_datapoints[2*s_idx+1]= d_datapoints[2*idx+1];

	__shared__ int s_clust_assn[TPB];
	s_clust_assn[s_idx] = d_clust_assn[idx];

	__syncthreads();

	// it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
    // COMMENT: this part could maybe be optimized? (could end up being 1 thread doing most of the computation)
	if(s_idx==0) {
		float b_clust_datapoint_sums[2*d_k]={0};
		int b_clust_sizes[d_k]={0};

		for(int j=0; j< blockDim.x; ++j) {
			int clust_id = s_clust_assn[j];
			b_clust_datapoint_sums[2*clust_id]+=s_datapoints[2*j];
            b_clust_datapoint_sums[2*clust_id+1]+=s_datapoints[2*j+1];
			b_clust_sizes[clust_id]+=1;
		}

		//Now we add the sums to the global centroids and add the counts to the global counts.
		for(int z=0; z < d_k; ++z) {
			atomicAdd(&d_centroids[2*z],b_clust_datapoint_sums[2*z]);
            atomicAdd(&d_centroids[2*z+1],b_clust_datapoint_sums[2*z+1]);
			atomicAdd(&d_clust_sizes[z],b_clust_sizes[z]);
		}
	}

	__syncthreads();

	//currently centroids are just sums, so divide by size to get actual centroids
	if(idx < d_k){
		d_centroids[idx] = d_centroids[idx]/d_clust_sizes[idx]; 
	}

}


int main()
{
    if (argc < 5) {
        std::cerr << "usage: k-means <data-file> <k> <n> [iterations]" << std::endl; // where n is the number of datapoints of the form (x1, x2)
        std::exit(EXIT_FAILURE);
    }

    const auto k = std::atoi(argv[3]);
    const auto n = std::atoi(argv[4]);
    const auto number_of_iterations = (argc == 6) ? std::atoi(argv[5]) : 300;

    // COMMENT: this way of storing data would limit the number of points in the dataset it is allowed to take
    // COMMENT: this also assumes knowledge of the number of points in the dataset
    // COMMENT: the dimensions of the points in the dataset is fixed at 2
    
    float *h_centroids = (float*)malloc(2*k*sizeof(float));
	float *h_datapoints = (float*)malloc(2*n*sizeof(float));
	int *h_clust_sizes = (int*)malloc(k*sizeof(int));

    std::ifstream stream(argv[2]);
    std::string line;
    int i = 0;
    while (std::getline(stream, line)) {
        std::istringstream line_stream(line);
        float x, y;
        uint16_t label;
        line_stream >> x >> y >> label;
        h_datapoints[2*i] = x;
        h_datapoints[2*i+1] = y;
    }

	//allocate memory on the device for the data points
	float *d_datapoints=0;
	//allocate memory on the device for the cluster assignments
	int *d_clust_assn = 0;
	//allocate memory on the device for the cluster centroids
	float *d_centroids = 0;
	//allocate memory on the device for the cluster sizes
	int *d_clust_sizes=0;
    //allocate memory on the device for number of clusters and number of data
    int *d_k=0; int *d_n=0;

	cudaMalloc(&d_datapoints, 2*n*sizeof(float));
	cudaMalloc(&d_clust_assn, n*sizeof(int));
	cudaMalloc(&d_centroids, 2*k*sizeof(float));
	cudaMalloc(&d_clust_sizes, k*sizeof(float));
    cudaMalloc(&d_k, sizeof(int));
	cudaMalloc(&d_n, sizeof(int));
    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

	srand(time(0));

	//initialize centroids
    for(int c=0;c<k;++c)
	{
		h_centroids[c*2]=(float) rand() / (double)RAND_MAX;
        h_centroids[c*2+1]=(float) rand() / (double)RAND_MAX;
		h_clust_sizes[c]=0;
	}

	cudaMemcpy(d_centroids,h_centroids, 2*k*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_datapoints,h_datapoints, 2*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_clust_sizes,h_clust_sizes, k*sizeof(int), cudaMemcpyHostToDevice);

	int cur_iter = 0;

    const auto start = std::chrono::high_resolution_clock::now();

	while(cur_iter < number_of_iterations)
	{
		//call cluster assignment kernel
		kMeansClusterAssignment<<<(n+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids,d_k,d_n);

		//reset centroids and cluster sizes (will be updated in the next kernel)
		cudaMemset(d_centroids,0.0,2*k*sizeof(float));
		cudaMemset(d_clust_sizes,0,2*K*sizeof(int));

		//call centroid update kernel
		kMeansCentroidUpdate<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids,d_clust_sizes,d_k,d_n);

		cur_iter+=1;
	}

    cudaMemcpy(h_centroids,d_centroids,2*k*sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(d_datapoints);
	cudaFree(d_clust_assn);
	cudaFree(d_centroids);
	cudaFree(d_clust_sizes);

	free(h_datapoints);
	free(h_clust_sizes);
    
    
    //code for timing
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    std::cerr << "Standard CUDA implementation Took: " << duration.count() << "s" << " for "<<h_x.size()<<" points."<<std::endl;
    std_time_used = duration.count();
    
    cudaFree(d_counts);

    std::vector<float> mean_x(k, 0);
    std::vector<float> mean_y(k, 0);
    cudaMemcpy(mean_x.data(), d_means.x, d_means.bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(mean_y.data(), d_means.y, d_means.bytes, cudaMemcpyDeviceToHost);

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
        fprintf(fp, "%0.6f %0.6f\n", h_centroids[2*i], h_centroids[2*i+1]);
    }
    free(h_centroids);
    fclose(fp);

}