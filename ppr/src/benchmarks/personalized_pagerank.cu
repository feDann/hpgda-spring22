// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <sstream>
#include "personalized_pagerank.cuh"
#include <math.h>
namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

//////////////////////////////
//////////////////////////////

//Parallel reduction

__device__ void warp_reduce(volatile double* input, int thread_id) {
    
    input[thread_id] += input[thread_id+32];
    input[thread_id] += input[thread_id+16];
    input[thread_id] += input[thread_id+8];
    input[thread_id] += input[thread_id+4];
    input[thread_id] += input[thread_id+2];
    input[thread_id] += input[thread_id+1];

}



__device__ void accumulate(double* input , int dim){
    int thread_id = threadIdx.x;
    if (dim > 32) {
        for (int i = dim/2; i > 32; i>>=1) {
            if (thread_id < i) {
                input[thread_id] += input[thread_id + i];
            }
            __syncthreads();
        }
    }

    if (thread_id < 32) {
        warp_reduce(input, thread_id);
    }

    __syncthreads();
}


__global__ void accumulate_global(double* input , int dim){
    int thread_id = threadIdx.x;

    extern __shared__ double shmem[];

    shmem[thread_id] = 0;
    for(int i = thread_id ; i < dim ; i+= blockDim.x ){
        shmem[thread_id] += input[i];
    }
    __syncthreads();

    
    for (int i = blockDim.x/2; i > 32; i>>=1) {
        if (thread_id < i) {
            shmem[thread_id] += shmem[thread_id + i];
        }
        __syncthreads();
    }

    if (thread_id < 32) {
        warp_reduce(shmem, thread_id);
    }

    __syncthreads();

    if(thread_id == 0){
        input[0] = shmem[0];
    }
}






// Write GPU kernel here!


__global__ void spmv_coo_gpu (const int* row_ids, const int* col_ids, const double* vals, const double* in_vec, double* out_vec , const int numVals) {
  for ( int i = threadIdx.x + blockIdx.x * blockDim.x ; i < numVals ; i += blockDim.x * gridDim.x ) {
    if ( i < numVals ) {
        atomicAdd(out_vec + row_ids[i], vals[i] * in_vec[col_ids[i]]);
    }
  }
}

__global__ void spmv_scoo_gpu(const int* col_ids, const int* row_ids, const double* vals, const int* idx, const double* in_vec, double* out_vec , const int numRows , const int numSlices, const int rows_per_slices, const int lane_size){

    int i = idx[blockIdx.x] + threadIdx.x; // 
    int end = idx[blockIdx.x + 1];
    
    int lane = threadIdx.x & (lane_size - 1);
    int row_lane = threadIdx.x/lane_size;

    int limit = ((blockIdx.x == numSlices - 1) ? ((numRows -1) % rows_per_slices) +1 : rows_per_slices);
    
    extern __shared__ double shrd_mem[];

    for(int index = row_lane; index < limit; index += (blockDim.x + lane_size -1) / lane_size){
        shrd_mem[index*lane_size + lane] = 0;
    }

    __syncthreads();

    while (i < end){
        int col = col_ids[i];
        int row = row_ids[i] - blockIdx.x * rows_per_slices;
        double val = vals[i];

        double res = in_vec[col] * val;

        atomicAdd(&shrd_mem[row*lane_size + lane], res);
        i+= blockDim.x;
    }

    __syncthreads();

    for (int index=row_lane; index<limit; index+=(blockDim.x + lane_size -1) / lane_size) { 
        volatile double *psdata = &shrd_mem[index*lane_size];
        int tid = (lane+index) & (lane_size - 1);

        if (lane_size>128 && lane<128) psdata[tid]+=psdata[(tid+128) & (lane_size-1)]; __syncthreads();
        if (lane_size>64 && lane<64) psdata[tid]+=psdata[(tid+64) & (lane_size-1)]; __syncthreads();
        if (lane_size>32 && lane<32) psdata[tid]+=psdata[(tid+32) & (lane_size-1)]; __syncthreads();

        if (lane_size>16 && lane<16) psdata[tid]+=psdata[( tid+16 ) & (lane_size-1)];
        if (lane_size>8 && lane<8) psdata[tid]+=psdata[( tid+8 ) & (lane_size-1)];
        if (lane_size>4 && lane<4) psdata[tid]+=psdata[( tid+4 ) & (lane_size-1)];
        if (lane_size>2 && lane<2) psdata[tid]+=psdata[( tid+2 ) & (lane_size-1)];
        if (lane_size>1 && lane<1) psdata[tid]+=psdata[( tid+1 ) & (lane_size-1)];
    }

    __syncthreads();

    int actual_row = blockIdx.x * rows_per_slices;

    for (int index = threadIdx.x ; index < limit ; index+= blockDim.x ){
        out_vec[actual_row + index] = shrd_mem[index*lane_size + lane];
    }


}

/*** Global Reduction Kernel  ***/

__global__ void reduce_global(double* partial , const int dim){
    
    unsigned threadId = threadIdx.x;
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double localVars[];

    localVars[threadId] = 0;

    __syncthreads();
    if(threadId < blockDim.x / 2 ){
        localVars[threadId] = partial[index] + partial[index+blockDim.x/2];
    }
    __syncthreads();

    for (unsigned i = blockDim.x / 4; i > 32; i >>= 1)
    {
        if (threadId < i)
        {
            localVars[threadId] += localVars[threadId + i];
        }
        __syncthreads();
    }

    if(threadId<32)
        warp_reduce(localVars,threadId);
    __syncthreads();

    if(threadId==0)
        partial[blockIdx.x] = localVars[threadId];
    __syncthreads();
}



/*** Dot Product Kernels  ***/
__global__ void dot_product_gpu (const int* vec1 , const double* vec2 , const int numVals , double* result){
    for(int i = threadIdx.x + blockIdx.x * blockDim.x ; i < numVals ; i += blockDim.x * gridDim.x ){
        if(i < numVals ){
            atomicAdd(result , vec1[i] * vec2[i]);
        }
    }

}

__global__ void dot_product_gpu_with_reduction (const int* vec1 , const double* vec2 , const int numVals , double* result) {
    extern __shared__ double tmp_res[];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numVals) {
        tmp_res[threadIdx.x] = vec1[id] * vec2[id];
    }
    __syncthreads();
    accumulate(tmp_res, blockDim.x);
    if (threadIdx.x == 0) {
        atomicAdd(result, tmp_res[0]);
    }
}

__global__ void dot_product_gpu_with_global_reduction (const int* vec1 , const double* vec2 , const int numVals , double* result) {
    extern __shared__ double tmp_res[];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numVals) {
        tmp_res[threadIdx.x] = vec1[id] * vec2[id];
    }
    __syncthreads();
    accumulate(tmp_res, blockDim.x);
    if (threadIdx.x == 0) {
        result[blockIdx.x] = tmp_res[0];
    }
}




/*** AXPB Personalized Kernels***/

__global__ void  axpb_personalized_gpu(const double alpha , const double* prTmp, const double beta, const int personalizationVertex , double* result , const int numVals){
    double oneMinusalpha = 1 - alpha;
    for(int i = threadIdx.x + blockIdx.x * blockDim.x ; i < numVals ; i += blockDim.x * gridDim.x ){
        if(i < numVals){
            result[i] = alpha * prTmp[i] + beta + ((personalizationVertex == i) ? oneMinusalpha : 0.0);
        }
    }
}


/*** Euclidean Distance Kernels ***/

__global__ void euclidean_distance_gpu(const double* pr , const double* prTmp , double* err , const int numVals){
        for(int i = threadIdx.x + blockIdx.x * blockDim.x ; i < numVals ; i += blockDim.x * gridDim.x ){
            if(i < numVals){
                double tmp = pr[i] - prTmp[i];
                atomicAdd(err, tmp*tmp);
            }
        }
}

__global__ void euclidean_distance_gpu_with_reduction (const double* pr , const double* prTmp , double* err , const int numVals) {
    extern __shared__ double tmp_res[];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numVals) {
        double tmp = pr[id] - prTmp[id];
        tmp_res[threadIdx.x] = tmp*tmp;
    }
    __syncthreads();
    accumulate(tmp_res, blockDim.x);
    if (threadIdx.x == 0) {
        atomicAdd(err, tmp_res[0]);
    }
}

__global__ void euclidean_distance_gpu_with_global_reduction (const double* pr , const double* prTmp , double* err , const int numVals) {
    extern __shared__ double tmp_res[];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numVals) {
        double tmp = pr[id] - prTmp[id];
        tmp_res[threadIdx.x] = tmp*tmp;
    }
    __syncthreads();
    accumulate(tmp_res, blockDim.x);
    if (threadIdx.x == 0) {
        err[blockIdx.x] = tmp_res[0];
    }
}



//////////////////////////////
//////////////////////////////

// CPU Utility functions;

int partition(std::vector<int>& col,std::vector<int>& row,std::vector<double>& vals, int start, int end){

	int pivot = col[start];

	int count = 0;
	for (int i = start + 1; i <= end; i++) {
		if (col[i] <= pivot)
			count++;
	}

	// Giving pivot element its correct position
	int pivotIndex = start + count;
	std::swap(col[pivotIndex], col[start]);
	std::swap(row[pivotIndex], row[start]);
	std::swap(vals[pivotIndex], vals[start]);

	// Sorting left and right parts of the pivot element
	int i = start, j = end;

	while (i < pivotIndex && j > pivotIndex) {

		while (col[i] <= pivot) {
			i++;
		}

		while (col[j] > pivot) {
			j--;
		}

		if (i < pivotIndex && j > pivotIndex) {
			std::swap(col[i], col[j]);
			std::swap(row[i], row[j]);
			std::swap(vals[i], vals[j]);
            i++;
            j--;
		}
	}

	return pivotIndex;
}

void quickSort(std::vector<int>& col ,std::vector<int>& row ,std::vector<double>& vals , int start, int end){

	// base case
	if (start >= end)
		return;

	// partitioning the array
	int p = partition(col,row,vals, start, end);

	// Sorting the left part
	quickSort(col,row,vals, start, p - 1);

	// Sorting the right part
	quickSort(col,row,vals, p + 1, end);
}


void PersonalizedPageRank::sort_scoo(){
    for(int i = 0; i < s_idx.size()-1; i++){
        int start = s_idx[i];
        int end = s_idx[i+1];

        quickSort(s_x, s_y, s_val , start , end-1);

    }
}



//transform from coo to scoo,
void PersonalizedPageRank::coo_to_scoo(int slice_size){
    s_x.resize(x.size());
    s_y.resize(y.size());
    s_val.resize(val.size());

    int ptr = 0;
    s_idx.resize((V+slice_size -1) / slice_size + 1);
    s_idx[0] = 0; //first s_idx is always 0;

    for(int i = 0; i < (V+slice_size -1) / slice_size; i++){
        for (int j = 0 ; j < E ; j++){
            if(x[j]>=(i * slice_size) && x[j] < (i*slice_size + slice_size)){
                s_x[ptr] = y[j];
                s_y[ptr] = x[j];
                s_val[ptr] = val[j];
                ptr++;
            }
            s_idx[i+1] = ptr;
        }
    }
}




// Read the input graph and initialize it;
void PersonalizedPageRank::initialize_graph() {
    

    // Read the graph from an MTX file;
    int num_rows = 0;
    int num_columns = 0;
    read_mtx(graph_file_path.c_str(), &x, &y, &val,
        &num_rows, &num_columns, &E, // Store the number of vertices (row and columns must be the same value), and edges;
        true,                        // If true, read edges TRANSPOSED, i.e. edge (2, 3) is loaded as (3, 2). We set this true as it simplifies the PPR computation;
        false,                       // If true, read the third column of the matrix file. If false, set all values to 1 (this is what you want when reading a graph topology);
        debug,                 
        false,                       // MTX files use indices starting from 1. If for whatever reason your MTX files uses indices that start from 0, set zero_indexed_file=true;
        true                         // If true, sort the edges in (x, y) order. If you have a sorted MTX file, turn this to false to make loading faster;
    );
    if (num_rows != num_columns) {
        if (debug) std::cout << "error, the matrix is not squared, rows=" << num_rows << ", columns=" << num_columns << std::endl;
        exit(-1);
    } else {
        V = num_rows;
    }
    if (debug) std::cout << "loaded graph, |V|=" << V << ", |E|=" << E << std::endl;

    // Compute the dangling vector. A vertex is not dangling if it has axleast 1 outgoing edge;
    dangling.resize(V);
    std::fill(dangling.begin(), dangling.end(), 1);  // Initially assume all vertices to be dangling;
    for (int i = 0; i < E; i++) {
        // Ignore self-loops, a vertex is still dangling if it has only self-loops;
        if (x[i] != y[i]) dangling[y[i]] = 0;
    }
    // Initialize the CPU PageRank vector;
    pr.resize(V);
    pr_golden.resize(V);
    // Initialize the value vector of the graph (1 / outdegree of each vertex).
    // Count how many edges start in each vertex (here, the source vertex is y as the matrix is transposed);
    int *outdegree = (int *) calloc(V, sizeof(int));
    for (int i = 0; i < E; i++) {
        outdegree[y[i]]++;
    }
    // Divide each edge value by the outdegree of the source vertex;
    for (int i = 0; i < E; i++) {
        val[i] = 1.0 / outdegree[y[i]];  
    }
    free(outdegree);
}

//////////////////////////////
//////////////////////////////

// Allocate data on the CPU and GPU;
void PersonalizedPageRank::alloc() {
    // Load the input graph and preprocess it;
    initialize_graph();

    //convert coo to scoo
    if( implementation > 1 ){
        int dev = 0; //id of the GPU
        cudaDeviceProp devProp;// is a C struct containing all the infos
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&devProp, dev);

        shared_memory = devProp.sharedMemPerBlock;
        rows_per_slice = shared_memory/sizeof(double)/lane_size;
        num_slices = (V +  rows_per_slice -1) / rows_per_slice;

        if(debug) std::cout << "Shared Memory: " << shared_memory << std::endl;
        if(debug) std::cout << "Rows Per Slices: " << rows_per_slice << std::endl;
        if(debug) std::cout << "Num Slices: " << num_slices << std::endl;

        if(debug) std::cout<< "coo to scoo conversion started" << std::endl;
        coo_to_scoo(rows_per_slice);
        if(debug) std::cout<< "coo to scoo conversion ended" << std::endl;

        if(debug) std::cout<< "slices sorting started" << std::endl;
        sort_scoo();
        if(debug) std::cout<< "slices sorting ended" << std::endl;

        CHECK(cudaMalloc(&d_idx, sizeof(int) * s_idx.size()));
        CHECK(cudaMemcpy(d_idx , s_idx.data(), sizeof(int) * s_idx.size(), cudaMemcpyHostToDevice));

    }
    int size_of_glob = (E + block_size -1)/block_size;
    size_of_partials = std::pow(block_size , std::ceil(std::log(size_of_glob)/std::log(block_size)));
    // size_of_partials = size_of_glob;

    if(implementation > 2){
        cudaStreamCreate(&stream_1);
        cudaStreamCreate(&stream_2);
        cudaStreamCreate(&stream_3);
    }

    // Allocate any GPU data here;
    CHECK(cudaMalloc(&d_x , sizeof(int) * E););
    CHECK(cudaMalloc(&d_y , sizeof(int) * E););
    CHECK(cudaMalloc(&d_val , sizeof(double) * E););
    CHECK(cudaMalloc(&d_dangling ,sizeof(int) * V););
    CHECK(cudaMalloc(&d_danglingFactor ,(implementation> 2 ? size_of_partials : 1)*  sizeof(double)););
    CHECK(cudaMalloc(&d_pr , sizeof(double) * V););
    CHECK(cudaMalloc(&d_prTmp , sizeof(double) * V););
    CHECK(cudaMalloc(&d_err ,sizeof(double)););

    CHECK(cudaMemcpy(d_x , implementation > 1 ? s_x.data() :x.data() , sizeof(int) * E , cudaMemcpyHostToDevice););
    CHECK(cudaMemcpy(d_y , implementation > 1 ? s_y.data() : y.data() , sizeof(int) * E , cudaMemcpyHostToDevice););
    CHECK(cudaMemcpy(d_val , implementation > 1 ? s_val.data() : val.data() , sizeof(double) * E , cudaMemcpyHostToDevice););
    CHECK(cudaMemcpy(d_dangling , dangling.data() , sizeof(int) * V , cudaMemcpyHostToDevice););


}

// Initialize data;
void PersonalizedPageRank::init() {
    // Do any additional CPU or GPU setup here;
    blockNums = (V + block_size -1)/block_size;
    threadsPerBlockNums = block_size;
}

// Reset the state of the computation after every iteration.
// Reset the result, and transfer data to the GPU if necessary;
void PersonalizedPageRank::reset() {
   // Reset the PageRank vector (uniform initialization, 1 / V for each vertex);
   std::fill(pr.begin(), pr.end(), 1.0 / V); 
   // Generate a new personalization vertex for this iteration;
   personalization_vertex = rand() % V; 
   if (debug) std::cout << "personalization vertex=" << personalization_vertex << std::endl;

   // Do any GPU reset here, and also transfer data to the GPU;
    CHECK(cudaMemcpy(d_pr , pr.data() , sizeof(double) * V , cudaMemcpyHostToDevice););


}

void swap(double * &a , double * &b){
    double * tmp = a;
    a = b;
    b = tmp;
}


void PersonalizedPageRank::ppr_0 (int iter) {
    auto start_tmp = clock_type::now();

    // Do the GPU computation here, and also transfer results to the CPU;
    int numIter = 0;
    bool converged = false;

    dim3 blocks(blockNums , 1 , 1);
    dim3 threads(threadsPerBlockNums, 1 , 1);
    dim3 blocks_spmv((E + block_size -1)/block_size , 1 , 1);

    while(numIter < max_iterations && !converged ){
        double danglingFactor;
        double err;

        CHECK(cudaMemset(d_prTmp , 0.0 , sizeof(double)*V););
        CHECK(cudaMemset(d_err , 0.0 , sizeof(double)););
        CHECK(cudaMemset(d_danglingFactor , 0.0 , sizeof(double)););
        
        spmv_coo_gpu<<<blocks_spmv , threads>>>(d_x, d_y, d_val ,d_pr , d_prTmp ,E);
        CHECK_KERNELCALL()


        dot_product_gpu<<<blocks , threads>>>(d_dangling , d_pr , V ,  d_danglingFactor);
        cudaMemcpy(&danglingFactor , d_danglingFactor , sizeof(double) , cudaMemcpyDeviceToHost);
        CHECK_KERNELCALL()
        
        axpb_personalized_gpu<<<blocks , threads>>>(alpha , d_prTmp , alpha * danglingFactor / V , personalization_vertex , d_prTmp , V);
        CHECK_KERNELCALL()

        euclidean_distance_gpu<<<blocks , threads>>>(d_pr, d_prTmp, d_err , V);
        cudaMemcpy(&err , d_err , sizeof(double) , cudaMemcpyDeviceToHost);
        CHECK_KERNELCALL()

        err = std::sqrt(err);
        converged = err <= convergence_threshold;

        CHECK(cudaMemcpy(d_pr , d_prTmp , sizeof(double)*V , cudaMemcpyDeviceToDevice));


        numIter++;

    }

    if (debug) {
        // Synchronize computation by hand to measure GPU exec. time;
        cudaDeviceSynchronize();
        auto end_tmp = clock_type::now();
        auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        std::cout << "  pure GPU execution(" << iter << ")=" << double(exec_time) / 1000 << " ms, " << (sizeof(double) * V / (exec_time * 1e3)) << " GB/s" << std::endl;
    }

    CHECK(cudaMemcpy(pr.data() , d_pr , sizeof(double) * V , cudaMemcpyDeviceToHost));
}

void PersonalizedPageRank::ppr_1(int iter) {
    auto start_tmp = clock_type::now();

    // Do the GPU computation here, and also transfer results to the CPU;
    int numIter = 0;
    bool converged = false;

    dim3 blocks(blockNums , 1 , 1);
    dim3 threads(threadsPerBlockNums, 1 , 1);
    dim3 blocks_spmv((E + block_size -1)/block_size , 1 , 1);

    while(numIter < max_iterations && !converged ){
        double danglingFactor;
        double err;

        CHECK(cudaMemset(d_prTmp , 0.0 , sizeof(double)*V););
        CHECK(cudaMemset(d_err , 0.0 , sizeof(double)););
        CHECK(cudaMemset(d_danglingFactor , 0.0 , sizeof(double)););
        
        spmv_coo_gpu<<<blocks_spmv , threads>>>(d_x, d_y, d_val ,d_pr , d_prTmp ,E);
        CHECK_KERNELCALL()


        dot_product_gpu_with_reduction<<<blocks , threads, block_size * sizeof(double)>>>(d_dangling , d_pr , V ,  d_danglingFactor);
        cudaMemcpy(&danglingFactor , d_danglingFactor , sizeof(double) , cudaMemcpyDeviceToHost);
        CHECK_KERNELCALL()
        
        axpb_personalized_gpu<<<blocks , threads>>>(alpha , d_prTmp , alpha * danglingFactor / V , personalization_vertex , d_prTmp , V);
        CHECK_KERNELCALL()

        euclidean_distance_gpu_with_reduction<<<blocks , threads, block_size * sizeof(double)>>>(d_pr, d_prTmp, d_err , V);
        cudaMemcpy(&err , d_err , sizeof(double) , cudaMemcpyDeviceToHost);
        CHECK_KERNELCALL()

        err = std::sqrt(err);
        converged = err <= convergence_threshold;

        CHECK(cudaMemcpy(d_pr , d_prTmp , sizeof(double)*V , cudaMemcpyDeviceToDevice));


        numIter++;

    }

    if (debug) {
        // Synchronize computation by hand to measure GPU exec. time;
        cudaDeviceSynchronize();
        auto end_tmp = clock_type::now();
        auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        std::cout << "  pure GPU execution(" << iter << ")=" << double(exec_time) / 1000 << " ms, " << (sizeof(double) * V / (exec_time * 1e3)) << " GB/s" << std::endl;
    }

    CHECK(cudaMemcpy(pr.data() , d_pr , sizeof(double) * V , cudaMemcpyDeviceToHost));

}


void PersonalizedPageRank::ppr_2(int iter) {
    auto start_tmp = clock_type::now();

    // Do the GPU computation here, and also transfer results to the CPU;
    int numIter = 0;
    bool converged = false;

    dim3 blocks(blockNums , 1 , 1);
    dim3 threads(threadsPerBlockNums, 1 , 1);
    dim3 blocks_spmv((E + block_size -1)/block_size , 1 , 1);

    while(numIter < max_iterations && !converged ){
        double danglingFactor;
        double err;

        CHECK(cudaMemset(d_prTmp , 0.0 , sizeof(double)*V););
        CHECK(cudaMemset(d_err , 0.0 , sizeof(double)););
        CHECK(cudaMemset(d_danglingFactor , 0.0 , sizeof(double)););
        
        spmv_scoo_gpu<<<num_slices , block_size , rows_per_slice * lane_size * sizeof(double)>>>(d_x, d_y, d_val, d_idx ,d_pr , d_prTmp ,V, num_slices, rows_per_slice, lane_size);
        CHECK_KERNELCALL()


        dot_product_gpu_with_reduction<<<blocks , threads, block_size * sizeof(double)>>>(d_dangling , d_pr , V ,  d_danglingFactor);
        cudaMemcpy(&danglingFactor , d_danglingFactor , sizeof(double) , cudaMemcpyDeviceToHost);
        CHECK_KERNELCALL()
        
        axpb_personalized_gpu<<<blocks , threads>>>(alpha , d_prTmp , alpha * danglingFactor / V , personalization_vertex , d_prTmp , V);
        CHECK_KERNELCALL()

        euclidean_distance_gpu_with_reduction<<<blocks , threads, block_size * sizeof(double)>>>(d_pr, d_prTmp, d_err , V);
        cudaMemcpy(&err , d_err , sizeof(double) , cudaMemcpyDeviceToHost);
        CHECK_KERNELCALL()

        err = std::sqrt(err);
        converged = err <= convergence_threshold;

        CHECK(cudaMemcpy(d_pr , d_prTmp , sizeof(double)*V , cudaMemcpyDeviceToDevice));


        numIter++;

    }

    if (debug) {
        // Synchronize computation by hand to measure GPU exec. time;
        cudaDeviceSynchronize();
        auto end_tmp = clock_type::now();
        auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        std::cout << "  pure GPU execution(" << iter << ")=" << double(exec_time) / 1000 << " ms, " << (sizeof(double) * V / (exec_time * 1e3)) << " GB/s" << std::endl;
    }

    CHECK(cudaMemcpy(pr.data() , d_pr , sizeof(double) * V , cudaMemcpyDeviceToHost));

}



void PersonalizedPageRank::ppr_3(int iter) {
    auto start_tmp = clock_type::now();

    // Do the GPU computation here, and also transfer results to the CPU;
    int numIter = 0;
    bool converged = false;


    dim3 blocks(blockNums , 1 , 1);
    dim3 threads(threadsPerBlockNums, 1 , 1);
    dim3 blocks_spmv((E + block_size -1)/block_size , 1 , 1);

    while(numIter < max_iterations && !converged ){
        double danglingFactor;
        double err;

        cudaMemsetAsync(d_prTmp,0,V * sizeof(double),stream_1);
        cudaMemsetAsync(d_danglingFactor,0,size_of_partials * sizeof(double),stream_3);
        cudaMemsetAsync(d_err,0,sizeof(double),stream_2);
        
        cudaDeviceSynchronize();
        spmv_scoo_gpu<<<num_slices , block_size , rows_per_slice * lane_size * sizeof(double), stream_2>>>(d_x, d_y, d_val, d_idx ,d_pr , d_prTmp ,V, num_slices, rows_per_slice, lane_size);

        dot_product_gpu_with_global_reduction<<<blocks , threads, block_size * sizeof(double), stream_1>>>(d_dangling , d_pr , V ,  d_danglingFactor);

        for(int i = size_of_partials/(block_size); i > 0; i/=block_size){
            reduce_global<<<i, block_size, block_size*sizeof(double), stream_1>>>(d_danglingFactor,i*block_size);
        }
        // accumulate_global<<<1, block_size, block_size*sizeof(double) , stream_1>>>(d_danglingFactor ,(V + block_size -1)/block_size );
        cudaMemcpy(&danglingFactor , d_danglingFactor , sizeof(double) , cudaMemcpyDeviceToHost);
        
        axpb_personalized_gpu<<<blocks , threads>>>(alpha , d_prTmp , alpha * danglingFactor / V , personalization_vertex , d_prTmp , V);

        euclidean_distance_gpu_with_reduction<<<blocks , threads, block_size * sizeof(double)>>>(d_pr, d_prTmp, d_err , V);

        cudaMemcpy(&err , d_err , sizeof(double) , cudaMemcpyDeviceToHost);

        err = std::sqrt(err);
        converged = err <= convergence_threshold;

        swap(d_pr , d_prTmp);

        numIter++;

    }

    if (debug) {
        // Synchronize computation by hand to measure GPU exec. time;
        cudaDeviceSynchronize();
        auto end_tmp = clock_type::now();
        auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        std::cout << "  pure GPU execution(" << iter << ")=" << double(exec_time) / 1000 << " ms, " << (sizeof(double) * V / (exec_time * 1e3)) << " GB/s" << std::endl;
    }

    CHECK(cudaMemcpy(pr.data() , d_pr , sizeof(double) * V , cudaMemcpyDeviceToHost));

}




void PersonalizedPageRank::execute(int iter) {
    
    switch (implementation)
    {
    case 0:
        ppr_0(iter);
        break;
    case 1:
        ppr_1(iter);
        break;   
    case 2:
        ppr_2(iter);
        break;
    case 3:
        ppr_3(iter);
        break;
    default:
        break;
    }

}

void PersonalizedPageRank::cpu_validation(int iter) {

    // Reset the CPU PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr_golden.begin(), pr_golden.end(), 1.0 / V);

    // Do Personalized PageRank on CPU;
    auto start_tmp = clock_type::now();
    personalized_pagerank_cpu(x.data(), y.data(), val.data(), V, E, pr_golden.data(), dangling.data(), personalization_vertex, alpha, 1e-6, 100);
    auto end_tmp = clock_type::now();
    auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
    if(debug) std::cout << "exec time CPU=" << double(exec_time) / 1000 << " ms" << std::endl;

    // Obtain the vertices with highest PPR value;
    std::vector<std::pair<int, double>> sorted_pr_tuples = sort_pr(pr.data(), V);
    std::vector<std::pair<int, double>> sorted_pr_golden_tuples = sort_pr(pr_golden.data(), V);

    // Check how many of the correct top-20 PPR vertices are retrieved by the GPU;
    std::set<int> top_pr_indices;
    std::set<int> top_pr_golden_indices;
    int old_precision = std::cout.precision();
    std::cout.precision(4);
    int topk = std::min(V, topk_vertices);
    for (int i = 0; i < topk; i++) {
        int pr_id_gpu = sorted_pr_tuples[i].first;
        int pr_id_cpu = sorted_pr_golden_tuples[i].first;
        top_pr_indices.insert(pr_id_gpu);
        top_pr_golden_indices.insert(pr_id_cpu);
        if (debug) {
            double pr_val_gpu = sorted_pr_tuples[i].second;
            double pr_val_cpu = sorted_pr_golden_tuples[i].second;
            if (pr_id_gpu != pr_id_cpu) {
                std::cout << "* error in rank! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            } else if (std::abs(sorted_pr_tuples[i].second - sorted_pr_golden_tuples[i].second) > 1e-6) {
                std::cout << "* error in value! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            }
        }
    }
    std::cout.precision(old_precision);
    // Set intersection to find correctly retrieved vertices;
    std::vector<int> correctly_retrieved_vertices;
    set_intersection(top_pr_indices.begin(), top_pr_indices.end(), top_pr_golden_indices.begin(), top_pr_golden_indices.end(), std::back_inserter(correctly_retrieved_vertices));
    precision = double(correctly_retrieved_vertices.size()) / topk;
    if (debug) std::cout << "correctly retrived top-" << topk << " vertices=" << correctly_retrieved_vertices.size() << " (" << 100 * precision << "%)" << std::endl;
}

std::string PersonalizedPageRank::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(precision);
    } else {
        // Print the first few PageRank values (not sorted);
        std::ostringstream out;
        out.precision(3);
        out << "[";
        for (int i = 0; i < std::min(20, V); i++) {
            out << pr[i] << ", ";
        }
        out << "...]";
        return out.str();
    }
}

void PersonalizedPageRank::clean() {
    // Delete any GPU data or additional CPU data;
    cudaFree(&d_x);
    cudaFree(&d_y);
    cudaFree(&d_val);
    cudaFree(&d_dangling);
    cudaFree(&d_danglingFactor);
    cudaFree(&d_pr);
    cudaFree(&d_prTmp);
    cudaFree(&d_err);

    if(implementation > 3){
        cudaStreamDestroy(stream_1);
        cudaStreamDestroy(stream_2);
        cudaStreamDestroy(stream_3);
    }
}
