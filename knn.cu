#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"

// compute distance by cpu
float compute_distance (const float* ref, const float* query, int dim) {
	float ed = 0.0;
	for (int i = 0; i < dim; ++ i) {
		float temp = ref[i] - query[i];
		ed += temp * temp;
	}
	return (float)sqrt(ed);
}

// compute knn by cpu
void knn_cpu (const float* ref, int ref_num, const float* query, int query_num, int dim, int k, float* knn_dist, int* knn_index) {
    for (int i = 0; i < query_num; ++ i) {
		float* nowdist = knn_dist + k * i;
		int* nowindex = knn_index + k * i;
		nowindex[0] = 0;
		nowdist[0] = compute_distance(ref, query + dim * i, dim);

        for (int m = 1; m < ref_num; ++ m) {
            float curr_dist = compute_distance(ref + dim * m, query + dim * i, dim);
            int curr_index = m;
			if (m >= k && curr_dist >= nowdist[k - 1]) {
				continue;
			}
			int j = m > (k - 1) ? (k - 1) : m;
			while (j > 0 && nowdist[j-1] > curr_dist) {
				nowdist[j]  = nowdist[j-1];
				nowindex[j] = nowindex[j-1];
				--j;
			}
			nowdist[j]  = curr_dist;
			nowindex[j] = curr_index; 
        }
    }
}

// check answer is right or not
bool issame(const int* a, const int* b, int n){
	int count = 0;
    for (int i = 0; i < n; i ++) {
        if (a[i] != b[i]) {
			count ++;
			// printf("%d %d\n", a[i], b[i]);
			// return false;
		}
    }
	printf("\ndifferent indexes count: %d\n", count);
    return count == 0;
}

// simply compute distance between each ref and each query
__global__ void compute_distance_ref2query(const float* __restrict__ ref, const float* __restrict__ query, float* distance, int ref_num, int query_num, int dim) {
	int tx = blockDim.x * blockIdx.x + threadIdx.x, ty = blockDim.y * blockIdx.y + threadIdx.y;
	float ed = 0.0;
	if (tx < query_num && ty < ref_num) {
		for (int i = 0; i < dim; i ++) {
			float temp = query[tx * dim + i] - ref[ty * dim + i];
			ed += temp * temp;
		}
		distance[tx * ref_num + ty] = (float)sqrt(ed);
	}
}

// find nearest k neighbors using k-insertion sort
__global__ void find_nearest_k_neighbors(float* distance, int* idx, int ref_num, int query_num, int k) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < query_num) {
		float* dis_of_query = distance + tid * ref_num;
		int* idx_of_query = idx + tid * k;
		for (int i = 0; i < ref_num; i ++) {
			float curr_dist = dis_of_query[i];
			int curr_index = i;
			if (i >= k && curr_dist >= dis_of_query[k - 1]) {
				continue;
			}
			int j = i > (k - 1) ? (k - 1) : i;
			while (j > 0 && dis_of_query[j - 1] > curr_dist) {
				dis_of_query[j] = dis_of_query[j - 1];
				idx_of_query[j] = idx_of_query[j - 1];
				--j;
			}
			dis_of_query[j] = curr_dist;
			idx_of_query[j] = curr_index;
		}
	}
}

// baseline
void knn_v1(const float* ref, int ref_num, const float* query, int query_num, int dim, int k, float* knn_dist, int* knn_index) {
	float* ref_dev, * que_dev, * dist_dev;
	int* idx_dev;
	CHECK(cudaMalloc((void**)&ref_dev, sizeof(float) * ref_num * dim));
	CHECK(cudaMalloc((void**)&que_dev, sizeof(float) * query_num * dim));
	CHECK(cudaMalloc((void**)&dist_dev, sizeof(float) * query_num * ref_num));
	CHECK(cudaMalloc((void**)&idx_dev, sizeof(int) * query_num * k));

	CHECK(cudaMemcpy(ref_dev, ref, sizeof(float) * ref_num * dim, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(que_dev, query, sizeof(float) * query_num * dim, cudaMemcpyHostToDevice));
	
	dim3 block1(32, 32), grid1(divup(query_num, 32), divup(ref_num, 32));
	compute_distance_ref2query<<<grid1, block1>>>(ref_dev, que_dev, dist_dev, ref_num, query_num, dim);

	dim3 block2(512), grid2(divup(query_num, 512));
	find_nearest_k_neighbors <<<grid2, block2>>>(dist_dev, idx_dev, ref_num, query_num, k);

    for (int i = 0; i < query_num; ++ i) {
        CHECK(cudaMemcpy(knn_dist + i * k, dist_dev + i * ref_num, sizeof(float) * k, cudaMemcpyDeviceToHost));    
    }
	CHECK(cudaMemcpy(knn_index, idx_dev, sizeof(int) * query_num * k, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(ref_dev));
	CHECK(cudaFree(que_dev));
	CHECK(cudaFree(dist_dev));
	CHECK(cudaFree(idx_dev));
}

// shared memory to compute distance
__global__ void compute_distance_sm(const float* __restrict__ ref, const float* __restrict__ query, float* distance, int ref_num, int query_num, int dim) {
    int blockx = blockDim.x * blockIdx.x, blocky = blockDim.y * blockIdx.y;
    int threadx = threadIdx.x, thready = threadIdx.y;
    float ed = 0.0;
    __shared__ float sub_queries[32][32], sub_refs[32][32];
    for (int i = 0; i < dim; i += 32) {
        // put sub points set into shared memory
        sub_queries[threadx][thready] = (blockx + threadx < query_num && i + thready < dim) ? query[(blockx + threadx) * dim + (i + thready)] : 0;
        sub_refs[thready][threadx] = (blocky + thready < ref_num && i + threadx < dim) ? ref[(blocky + thready) * dim + (i + threadx)] : 0;
        
        __syncthreads();
        if(blockx + threadx < query_num && blocky + thready < ref_num) {
            for (int j = 0; j < 32; ++ j) {
                float temp = sub_queries[threadx][j] - sub_refs[thready][j];
                ed += temp * temp;
            }
        }
        __syncthreads();
    }
    if(blockx + threadx < query_num && blocky + thready < ref_num) distance[(blockx + threadx) * ref_num + (blocky + thready)] = (float)sqrt(ed);
}

// baseline + shared memory to compute distance
void knn_v2(const float* ref, int ref_num, const float* query, int query_num, int dim, int k, float* knn_dist, int* knn_index) {
    float* ref_dev, * que_dev, * dist_dev;
	int* idx_dev;
	CHECK(cudaMalloc((void**)&ref_dev, sizeof(float) * ref_num * dim));
	CHECK(cudaMalloc((void**)&que_dev, sizeof(float) * query_num * dim));
	CHECK(cudaMalloc((void**)&dist_dev, sizeof(float) * query_num * ref_num));
	CHECK(cudaMalloc((void**)&idx_dev, sizeof(int) * query_num * k));

	CHECK(cudaMemcpy(ref_dev, ref, sizeof(float) * ref_num * dim, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(que_dev, query, sizeof(float) * query_num * dim, cudaMemcpyHostToDevice));
	
	dim3 block1(32, 32), grid1(divup(query_num, 32), divup(ref_num, 32));
	compute_distance_sm<<<grid1, block1>>>(ref_dev, que_dev, dist_dev, ref_num, query_num, dim);

	dim3 block2(512), grid2(divup(query_num, 512));
	find_nearest_k_neighbors <<<grid2, block2>>>(dist_dev, idx_dev, ref_num, query_num, k);

    for (int i = 0; i < query_num; ++ i) {
        CHECK(cudaMemcpy(knn_dist + i * k, dist_dev + i * ref_num, sizeof(float) * k, cudaMemcpyDeviceToHost));    
    }
	CHECK(cudaMemcpy(knn_index, idx_dev, sizeof(int) * query_num * k, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(ref_dev));
	CHECK(cudaFree(que_dev));
	CHECK(cudaFree(dist_dev));
	CHECK(cudaFree(idx_dev));
}

// reduction without sorted knn
__device__ void reduction_operation_unsorted(float* des, int* idxdes, float* src, int idxsrc1, int idxsrc2, int k) {
    for (int i = idxsrc1; i < idxsrc2; ++ i) {
        int current_idx = i;
		float current_dis = src[current_idx];
        if ((i - idxsrc1) >= k && des[k - 1] <= current_dis) {
            continue;
        }
        int j = (i - idxsrc1) > (k - 1) ? (k - 1) : (i - idxsrc1);
        while(j > 0 && des[j - 1] > current_dis){
            des[j] = des[j - 1];
            idxdes[j] = idxdes[j - 1];
            -- j;
        }
        des[j] = current_dis;
        idxdes[j] = current_idx;
    }
}

// reduction with a sorted and an unsorted vector
// __device__ void reduction_operation_sort_unsort(float* des, int* idxdes, float *src, int idxsrc, int k) {
//     for (int i = 0; i < k; ++ i) {
//         float current_dis = src[i];
//         int current_idx = idxsrc + i;
//         if(des[k - 1] <= current_dis) continue;
//         int j = k - 1;
//         while(j > 0 && des[j - 1] > current_dis){
//             des[j] = des[j - 1];
//             idxdes[j] = idxdes[j - 1];
//             -- j;
//         }
//         des[j] = current_dis;
//         idxdes[j] = current_idx;
//     }
// }

// reduction sm to sm, no need to sort it, merge
__device__ void reduction_operation_sorted(float* des, int* idxdes, float* src, int* idxsrc, int k) {
    for (int i = 0; i < k; ++ i) {
        float current_dis = src[i];
        float current_idx = idxsrc[i];
        if(des[k - 1] <= current_dis) continue;
        int j = k - 1;
        while(j > 0 && des[j - 1] > current_dis){
            des[j] = des[j - 1];
            idxdes[j] = idxdes[j - 1];
            -- j;
        }
        des[j] = current_dis;
        idxdes[j] = current_idx;
    }
}

__device__ void warpReduce (float* sm, int* smidx, int tid, int k) {
	reduction_operation_sorted(sm + tid * k, smidx + tid * k, sm + (tid + 32) * k, smidx + (tid + 32) * k, k);
	if (tid < 16) reduction_operation_sorted(sm + tid * k, smidx + tid * k, sm + (tid + 16) * k, smidx + (tid + 16) * k, k);
	if (tid < 8) reduction_operation_sorted(sm + tid * k, smidx + tid * k, sm + (tid + 8) * k, smidx + (tid + 8) * k, k);
	if (tid < 4) reduction_operation_sorted(sm + tid * k, smidx + tid * k, sm + (tid + 4) * k, smidx + (tid + 4) * k, k);
	if (tid < 2) reduction_operation_sorted(sm + tid * k, smidx + tid * k, sm + (tid + 2) * k, smidx + (tid + 2) * k, k);
	if (tid < 1) reduction_operation_sorted(sm + tid * k, smidx + tid * k, sm + (tid + 1) * k, smidx + (tid + 1) * k, k);
}

// find knn by reduction
__global__ void reduction_find_kNN(float* distance, int* idx, int ref_num, int query_num, int k, int pre_reduction_step) {
    extern __shared__ float sm[];
	// bid is the query point index, tid is the batch index of distances
	int tid = threadIdx.x, bid = blockIdx.x, blocksize = blockDim.x;
    int* smidx = (int *)(sm + k * blocksize);
    float* dis_base = distance + bid * ref_num;
	int* idx_base = idx + bid * k;
    // control and load all batches to a query into sm in a block
    // predo: compare then load it - save shared memory and speed up
	if(tid < blocksize){
		int begin = tid * pre_reduction_step * k;
		int end = (tid != blocksize - 1) ? (tid + 1) * pre_reduction_step * k : ref_num;
		reduction_operation_unsorted(sm + tid * k, smidx + tid * k, dis_base, begin, end, k);
	}
    __syncthreads();

    // reduction
    for (int step = blocksize >> 1; step > 32; step >>= 1) {
        if (tid < step) {
            reduction_operation_sorted(sm + tid * k, smidx + tid * k, sm + (tid + step) * k, smidx + (tid + step) * k, k);
        }
        __syncthreads();
    }
    if (tid < 32) {
		warpReduce(sm, smidx, tid, k);
    }
	__syncthreads();

    if (tid < k) {
        // write back
        dis_base[tid] = sm[tid];
        idx_base[tid] = smidx[tid];
    }
}

// baseline + shared memory + reduction
void knn_v3(const float* ref, int ref_num, const float* query, int query_num, int dim, int k, float* knn_dist, int* knn_index) {
    float* ref_dev, * que_dev, * dist_dev;
	int* idx_dev;
	CHECK(cudaMalloc((void**)&ref_dev, sizeof(float) * ref_num * dim));
	CHECK(cudaMalloc((void**)&que_dev, sizeof(float) * query_num * dim));
	CHECK(cudaMalloc((void**)&dist_dev, sizeof(float) * query_num * ref_num));
	CHECK(cudaMalloc((void**)&idx_dev, sizeof(int) * query_num * k));

	CHECK(cudaMemcpy(ref_dev, ref, sizeof(float) * ref_num * dim, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(que_dev, query, sizeof(float) * query_num * dim, cudaMemcpyHostToDevice));
	
	dim3 block1(32, 32), grid1(divup(query_num, 32), divup(ref_num, 32));
	compute_distance_sm<<<grid1, block1>>>(ref_dev, que_dev, dist_dev, ref_num, query_num, dim);

	dim3 block2(256), grid2(query_num);
	int predo = ref_num /(256 * k);
	reduction_find_kNN<<<grid2, block2, sizeof(float) * k * 512>>>(dist_dev, idx_dev, ref_num, query_num, k, predo);

    for (int i = 0; i < query_num; ++ i) {
        CHECK(cudaMemcpy(knn_dist + i * k, dist_dev + i * ref_num, sizeof(float) * k, cudaMemcpyDeviceToHost));    
    }
	CHECK(cudaMemcpy(knn_index, idx_dev, sizeof(int) * query_num * k, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(ref_dev));
	CHECK(cudaFree(que_dev));
	CHECK(cudaFree(dist_dev));
	CHECK(cudaFree(idx_dev));
}

// baseline + shared memory + reduction + a better blocksize
void knn_v4(const float* ref, int ref_num, const float* query, int query_num, int dim, int k, float* knn_dist, int* knn_index) {
    float* ref_dev, * que_dev, * dist_dev;
	int* idx_dev;
	CHECK(cudaMalloc((void**)&ref_dev, sizeof(float) * ref_num * dim));
	CHECK(cudaMalloc((void**)&que_dev, sizeof(float) * query_num * dim));
	CHECK(cudaMalloc((void**)&dist_dev, sizeof(float) * query_num * ref_num));
	CHECK(cudaMalloc((void**)&idx_dev, sizeof(int) * query_num * k));

	CHECK(cudaMemcpy(ref_dev, ref, sizeof(float) * ref_num * dim, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(que_dev, query, sizeof(float) * query_num * dim, cudaMemcpyHostToDevice));
	
	dim3 block1(32, 32), grid1(divup(query_num, 32), divup(ref_num, 32));
	compute_distance_sm<<<grid1, block1>>>(ref_dev, que_dev, dist_dev, ref_num, query_num, dim);

	dim3 block2(64), grid2(query_num);
	int predo = ref_num / (64 * k);
	reduction_find_kNN<<<grid2, block2, sizeof(float) * k * 128>>>(dist_dev, idx_dev, ref_num, query_num, k, predo);

    for (int i = 0; i < query_num; ++ i) {
        CHECK(cudaMemcpy(knn_dist + i * k, dist_dev + i * ref_num, sizeof(float) * k, cudaMemcpyDeviceToHost));    
    }
	CHECK(cudaMemcpy(knn_index, idx_dev, sizeof(int) * query_num * k, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(ref_dev));
	CHECK(cudaFree(que_dev));
	CHECK(cudaFree(dist_dev));
	CHECK(cudaFree(idx_dev));
}

// initialize two points set
void initialize_data(float* ref, int ref_nb, float* query, int query_nb, int dim) {
	// Initialize random number generator
	srand(12345);

	// Generate random reference points
	for (int i = 0; i < ref_nb * dim; ++i) {
		ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
	}

	// Generate random query points
	for (int i = 0; i < query_nb * dim; ++i) {
		query[i] = 10. * (float)(rand() / (double)RAND_MAX);
	}
}


int main(void) {

	// Parameters
	const int ref_nb = 8192;
	const int query_nb = 512;
	const int dim = 128;
	const int k = 16;

	// Display
	printf("PARAMETERS\n");
	printf("- Number reference points : %d\n", ref_nb);
	printf("- Number query points     : %d\n", query_nb);
	printf("- Dimension of points     : %d\n", dim);
	printf("- Number of neighbors     : %d\n\n", k);

	// Allocate input points and output k-NN distances / indexes
	float* ref = (float*)malloc(ref_nb * dim * sizeof(float));
	float* query = (float*)malloc(query_nb * dim * sizeof(float));
	float* knn_dist = (float*)malloc(query_nb * k * sizeof(float));
	int* knn_index = (int*)malloc(query_nb * k * sizeof(int));
    int* knn_index2 = (int*)malloc(query_nb * k * sizeof(int));

	// Initialize reference and query points with random values
	initialize_data(ref, ref_nb, query, query_nb, dim);
    // warm up
	knn_v1(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index);


	clock_t start, end;

	start = clock();
	// Compute k-NN several times
	for (int i = 0; i < 20; ++i) {
		knn_v1(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index);
	}
	end = clock();
	double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
	// Test all k-NN functions
	printf("\nv1: %lf\n", elapsed_time / 20);

    start = clock();
	// Compute k-NN several times
	for (int i = 0; i < 20; ++i) {
		knn_v2(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index2);
	}
	end = clock();
	elapsed_time = double(end - start) / CLOCKS_PER_SEC;
	// Test all k-NN functions
	printf("\nv2: %lf\n", elapsed_time / 20);

    printf("\nVersion_1 and Version_2 have %s.\n\n", issame(knn_index, knn_index2, query_nb * k) ? "the same answer" : "different answers");

	start = clock();
	// Compute k-NN several times
	for (int i = 0; i < 20; ++i) {
		knn_v3(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index2);
	}
	end = clock();
	elapsed_time = double(end - start) / CLOCKS_PER_SEC;
	// Test all k-NN functions
	printf("\nv3: %lf\n", elapsed_time / 20);

    printf("\nVersion_1 and Version_3 have %s.\n\n", issame(knn_index, knn_index2, query_nb * k) ? "the same answer" : "different answers");

	start = clock();
	// Compute k-NN several times
	for (int i = 0; i < 20; ++i) {
		knn_v4(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index2);
	}
	end = clock();
	elapsed_time = double(end - start) / CLOCKS_PER_SEC;
	// Test all k-NN functions
	printf("\nv4: %lf\n", elapsed_time / 20);

    printf("\nVersion_1 and Version_4 have %s.\n\n", issame(knn_index, knn_index2, query_nb * k) ? "the same answer" : "different answers");

	start = clock();
	// Compute k-NN several times
	for (int i = 0; i < 2; ++i) {
		knn_cpu(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index2);
	}
	end = clock();
	elapsed_time = double(end - start) / CLOCKS_PER_SEC;
	// Test all k-NN functions
	printf("\ncpu: %lf\n", elapsed_time / 2);

    printf("\nVersion_1 and Version_CPU have %s.\n\n", issame(knn_index, knn_index2, query_nb * k) ? "the same answer" : "different answers");

	free(ref);
	free(query);
	free(knn_dist);
	free(knn_index);

	return 0;
}