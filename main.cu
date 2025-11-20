#include "secp256k1.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <fstream>
#include <stdint.h>
#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <inttypes.h>
#include <windows.h>
#include <bcrypt.h>
#include <chrono>
#pragma comment(lib, "bcrypt.lib")
#pragma once


#define NUM_HASH_FUNCTIONS 10


__device__ uint64_t d_bloom_size_bytes;

__device__ __host__ __forceinline__ uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

__device__ __host__ void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}

__device__ __host__ void hex_to_bigint(const char* hex_str, BigInt* bigint) {
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (hex_str[len] != '\0' && len < 64) len++;
    
    int word_idx = 0;
    int bit_offset = 0;
    
    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        uint8_t val = hex_char_to_byte(hex_str[i]);
        bigint->data[word_idx] |= ((uint32_t)val << bit_offset);
        bit_offset += 4;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

__device__ void bigint_to_hex(const BigInt* bigint, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int idx = 0;
    bool leading_zero = true;
    
    for (int i = 7; i >= 0; i--) {
        for (int j = 28; j >= 0; j -= 4) {
            uint8_t nibble = (bigint->data[i] >> j) & 0xF;
            if (nibble != 0 || !leading_zero || (i == 0 && j == 0)) {
                hex_str[idx++] = hex_chars[nibble];
                leading_zero = false;
            }
        }
    }
    
    if (idx == 0) {
        hex_str[idx++] = '0';
    }
    
    hex_str[idx] = '\0';
}

__device__ __forceinline__ void byte_to_hex(uint8_t byte, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    out[0] = hex_chars[(byte >> 4) & 0xF];
    out[1] = hex_chars[byte & 0xF];
}

__device__ void hash160_to_hex(const uint8_t *hash, char *out_hex) {
    const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 20; ++i) {
        out_hex[i * 2]     = hex_chars[hash[i] >> 4];
        out_hex[i * 2 + 1] = hex_chars[hash[i] & 0x0F];
    }
    out_hex[40] = '\0';
}

__device__ void bigint_increment(BigInt* a) {
    uint64_t carry = 1;
    for (int i = 0; i < 4; ++i) {
        uint64_t sum = a->data[i] + carry;
        carry = (sum < a->data[i]) ? 1 : 0;
        a->data[i] = sum;
        if (carry == 0) break;
    }
}


#define NUM_HASH_FUNCTIONS 10
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL
#define GOLDEN64   0x9e3779b97f4a7c13ULL

__device__ __forceinline__ uint64_t fnv1a_hash_cuda(const uint8_t* data, int len, uint64_t seed) {
    uint64_t hash = FNV_OFFSET ^ seed;
    for (int i = 0; i < len; i++) {
        hash ^= (uint64_t)data[i];
        hash *= FNV_PRIME;
        hash &= 0xFFFFFFFFFFFFFFFFULL; 
    }
    return hash;
}

__device__ __forceinline__ bool bloom_check(
    const uint8_t* bloom_filter,
    uint64_t bloom_size_bytes,
    const uint8_t* hash160
) {
    uint64_t bloom_bits = bloom_size_bytes * 8ULL;

    for (int i = 0; i < NUM_HASH_FUNCTIONS; i++) {
        uint64_t seed = (uint64_t)i * GOLDEN64;  
        uint64_t hv = fnv1a_hash_cuda(hash160, 20, seed);

        uint64_t bit_index = hv % bloom_bits;
        uint64_t byte_index = bit_index / 8;
        uint8_t bit_offset = bit_index % 8;

        
        if (!(bloom_filter[byte_index] & (1 << (7 - bit_offset)))) {
            return false;
        }
    }
    return true;
}


__device__ void generate_random_in_range(BigInt* result, curandStatePhilox4_32_10_t* state, 
                                         const BigInt* min_val, const BigInt* max_val) {
    BigInt range;
    bool borrow = false;
    
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t diff = (uint64_t)max_val->data[i] - (uint64_t)min_val->data[i] - (borrow ? 1 : 0);
        range.data[i] = (uint32_t)diff;
        borrow = (diff > 0xFFFFFFFFULL);
    }
    
    BigInt random;
    for (int w = 0; w < BIGINT_WORDS; w += 4) {
        uint4 r = curand4(state);
        if (w + 0 < BIGINT_WORDS) random.data[w + 0] = r.x;
        if (w + 1 < BIGINT_WORDS) random.data[w + 1] = r.y;
        if (w + 2 < BIGINT_WORDS) random.data[w + 2] = r.z;
        if (w + 3 < BIGINT_WORDS) random.data[w + 3] = r.w;
    }
    
    int highest_word = BIGINT_WORDS - 1;
    while (highest_word >= 0 && range.data[highest_word] == 0) {
        highest_word--;
    }
    
    if (highest_word >= 0) {
        uint32_t mask = range.data[highest_word];
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        
        asm volatile ("and.b32 %0, %1, %2;" 
                     : "=r"(random.data[highest_word]) 
                     : "r"(random.data[highest_word]), "r"(mask));
        
        for (int i = highest_word + 1; i < BIGINT_WORDS; ++i) {
            asm volatile ("mov.b32 %0, 0;" : "=r"(random.data[i]));
        }
        
        bool greater = false;
        for (int i = BIGINT_WORDS - 1; i >= 0; --i) {
            if (random.data[i] > range.data[i]) {
                greater = true;
                break;
            } else if (random.data[i] < range.data[i]) {
                break;
            }
        }
        
        if (greater) {
            for (int i = 0; i < BIGINT_WORDS; ++i) {
                uint32_t divisor = range.data[i] + 1;
                if (divisor != 0) {
                    asm volatile ("rem.u32 %0, %1, %2;" 
                                 : "=r"(random.data[i]) 
                                 : "r"(random.data[i]), "r"(divisor));
                }
            }
        }
    }
    
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint32_t r_word = random.data[i];
        uint32_t min_word = min_val->data[i];
        
        if (i == 0) {
            asm volatile ("add.cc.u32 %0, %1, %2;" 
                         : "=r"(result->data[0]) 
                         : "r"(r_word), "r"(min_word));
        } else if (i == BIGINT_WORDS - 1) {
            asm volatile ("addc.u32 %0, %1, %2;" 
                         : "=r"(result->data[i]) 
                         : "r"(r_word), "r"(min_word));
        } else {
            asm volatile ("addc.cc.u32 %0, %1, %2;" 
                         : "=r"(result->data[i]) 
                         : "r"(r_word), "r"(min_word));
        }
    }
}

__constant__ BigInt d_min_bigint;
__constant__ BigInt d_max_bigint;
__device__ volatile int g_found = 0;
__device__ char g_found_hex[65] = {0};
__device__ char g_found_hash160[41] = {0};
__device__ uint8_t g_found_type = 0; 

__global__ void start(uint64_t seed, uint8_t* bloom_filter, uint64_t bloom_size_bytes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);
    
    BigInt priv_base;
    BigInt priv_current;
    ECPointJac result_jac_batch[BATCH_SIZE];
    uint8_t hash160_compressed[BATCH_SIZE][20];
    uint8_t hash160_uncompressed[BATCH_SIZE][20];
    
    generate_random_in_range(&priv_base, &state, &d_min_bigint, &d_max_bigint);
    
    scalar_multiply_multi_base_jac(&result_jac_batch[0], &priv_base);
    
    for (int i = 1; i < BATCH_SIZE; ++i) {
        add_G_to_point_jac(&result_jac_batch[i], &result_jac_batch[i-1]);
    }
    
    
    jacobian_batch_to_hash160(result_jac_batch, hash160_compressed);
    jacobian_batch_to_hash160_uncompressed(result_jac_batch, hash160_uncompressed);
    
    
    for (int i = 0; i < BATCH_SIZE; ++i) {
        if (bloom_check(bloom_filter, bloom_size_bytes, hash160_compressed[i])) {
            if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                priv_current = priv_base;
                for (int j = 0; j < i; ++j) {
                    bigint_increment(&priv_current);
                }
                
                bigint_to_hex(&priv_current, g_found_hex);
                hash160_to_hex(hash160_compressed[i], g_found_hash160);
                g_found_type = 0; 
            }
            return; 
        }
    }
    
    
    for (int i = 0; i < BATCH_SIZE; ++i) {
        if (bloom_check(bloom_filter, bloom_size_bytes, hash160_uncompressed[i])) {
            if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                priv_current = priv_base;
                for (int j = 0; j < i; ++j) {
                    bigint_increment(&priv_current);
                }
                
                bigint_to_hex(&priv_current, g_found_hex);
                hash160_to_hex(hash160_uncompressed[i], g_found_hash160);
                g_found_type = 1; 
            }
            return; 
        }
    }
}

bool load_bloom_filter(const char* filename, uint8_t** bloom_filter_host, uint8_t** bloom_filter_device, size_t* bloom_size) {
    printf("Opening bloom filter file: %s\n", filename);
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open bloom filter file: " << filename << std::endl;
        std::cerr << "Please check if the file exists and is accessible." << std::endl;
        return false;
    }
    
    
    file.seekg(0, std::ios::end);
    *bloom_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (*bloom_size == 0) {
        std::cerr << "Error: Bloom filter file is empty!" << std::endl;
        file.close();
        return false;
    }
    
    printf("Bloom filter file size: %.2f MB (%.2f GB)\n", 
           (*bloom_size / (1024.0 * 1024.0)),
           (*bloom_size / (1024.0 * 1024.0 * 1024.0)));
    
    
    size_t free_mem, total_mem;
    cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (mem_err != cudaSuccess) {
        std::cerr << "Error getting GPU memory info: " << cudaGetErrorString(mem_err) << std::endl;
        file.close();
        return false;
    }
    
    printf("GPU Memory - Free: %.2f MB (%.2f GB), Total: %.2f MB (%.2f GB)\n", 
           (free_mem / (1024.0 * 1024.0)),
           (free_mem / (1024.0 * 1024.0 * 1024.0)),
           (total_mem / (1024.0 * 1024.0)),
           (total_mem / (1024.0 * 1024.0 * 1024.0)));
    
    if (*bloom_size > free_mem * 0.9) {  
        std::cerr << "Error: Bloom filter size (" << (*bloom_size / (1024.0 * 1024.0)) 
                  << " MB) exceeds available GPU memory (" << (free_mem / (1024.0 * 1024.0)) 
                  << " MB)" << std::endl;
        std::cerr << "Please use a smaller bloom filter or free up GPU memory." << std::endl;
        file.close();
        return false;
    }
    
    std::cout << "Allocating host memory..." << std::endl;
    
    
    try {
        *bloom_filter_host = new (std::nothrow) uint8_t[*bloom_size];
        if (*bloom_filter_host == nullptr) {
            std::cerr << "Error: Failed to allocate host memory for bloom filter!" << std::endl;
            std::cerr << "Requested: " << (*bloom_size / (1024.0 * 1024.0)) << " MB" << std::endl;
            file.close();
            return false;
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Failed to allocate host memory: " << e.what() << std::endl;
        file.close();
        return false;
    }
    
    std::cout << "Reading bloom filter from disk..." << std::endl;
    
    
    const size_t chunk_size = 100 * 1024 * 1024; 
    size_t bytes_read = 0;
    
    while (bytes_read < *bloom_size) {
        size_t to_read = std::min(chunk_size, *bloom_size - bytes_read);
        file.read(reinterpret_cast<char*>(*bloom_filter_host + bytes_read), to_read);
        
        if (!file) {
            std::cerr << "Error: Failed to read bloom filter file at byte " << bytes_read << std::endl;
            delete[] *bloom_filter_host;
            *bloom_filter_host = nullptr;
            file.close();
            return false;
        }
        
        bytes_read += to_read;
        
        if (*bloom_size > 100 * 1024 * 1024) { 
            printf("\rReading: %.1f%%", (bytes_read * 100.0) / *bloom_size);
            fflush(stdout);
        }
    }
    
    if (*bloom_size > 100 * 1024 * 1024) {
        printf("\n");
    }
    
    file.close();
    std::cout << "File read complete." << std::endl;
    
    
    std::cout << "Allocating GPU memory (" << (*bloom_size / (1024.0 * 1024.0)) << " MB)..." << std::endl;
    cudaError_t err = cudaMalloc(bloom_filter_device, *bloom_size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for bloom filter: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "Requested size: " << (*bloom_size / (1024.0 * 1024.0)) << " MB" << std::endl;
        delete[] *bloom_filter_host;
        *bloom_filter_host = nullptr;
        return false;
    }
    
    std::cout << "GPU memory allocated successfully." << std::endl;
    
    
    std::cout << "Copying bloom filter to GPU memory..." << std::endl;
    err = cudaMemcpy(*bloom_filter_device, *bloom_filter_host, *bloom_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying bloom filter to device: " << cudaGetErrorString(err) << std::endl;
        delete[] *bloom_filter_host;
        *bloom_filter_host = nullptr;
        cudaFree(*bloom_filter_device);
        *bloom_filter_device = nullptr;
        return false;
    }
    
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error during device synchronization: " << cudaGetErrorString(err) << std::endl;
        delete[] *bloom_filter_host;
        *bloom_filter_host = nullptr;
        cudaFree(*bloom_filter_device);
        *bloom_filter_device = nullptr;
        return false;
    }
    
    std::cout << "✓ Bloom filter successfully loaded into GPU memory!" << std::endl;
    std::cout << "  Size: " << (*bloom_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Bits: " << (*bloom_size * 8) << " (" << (*bloom_size * 8 / 1000000000.0) << " billion)" << std::endl;
    
    return true;
}

bool run_with_bloom_filter(const char* min, const char* max, const char* bloom_file, 
                           int blocks, int threads, int device_id) {
    BigInt min_bigint, max_bigint;
    hex_to_bigint(min, &min_bigint);
    hex_to_bigint(max, &max_bigint);
    
    cudaMemcpyToSymbol(d_min_bigint, &min_bigint, sizeof(BigInt));
    cudaMemcpyToSymbol(d_max_bigint, &max_bigint, sizeof(BigInt));
    
    
    uint8_t *bloom_filter_host = nullptr;
    uint8_t *bloom_filter_device = nullptr;
    size_t bloom_size = 0;
    
    if (!load_bloom_filter(bloom_file, &bloom_filter_host, &bloom_filter_device, &bloom_size)) {
        return false;
    }
    
    
    uint64_t bloom_size_u64 = bloom_size;
    cudaMemcpyToSymbol(d_bloom_size_bytes, &bloom_size_u64, sizeof(uint64_t));
    
    int total_threads = blocks * threads;
    int found_flag;
    
    uint64_t keys_per_kernel = (uint64_t)blocks * threads * BATCH_SIZE;
    
    printf("\n=== Search Configuration ===\n");
    printf("Range Min: %s\n", min);
    printf("Range Max: %s\n", max);
    printf("Bloom filter: %s (%.2f MB, stored in GPU memory)\n", bloom_file, bloom_size / (1024.0 * 1024.0));
    printf("Blocks: %d, Threads: %d, Batch size: %d\n", blocks, threads, BATCH_SIZE);
    printf("Total GPU threads: %d\n", total_threads);
    printf("Keys per kernel: %llu\n", (unsigned long long)keys_per_kernel);
    printf("===========================\n\n");
    
    uint64_t seed;
    uint64_t total_keys_checked = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print_time = start_time;
    BCryptGenRandom(NULL, (PUCHAR)&seed, sizeof(seed), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    
    printf("Starting search...\n");
    
    while(true) {
        start<<<blocks, threads>>>(seed, bloom_filter_device, bloom_size);
        
        seed += 1;
        total_keys_checked += keys_per_kernel;
        
        cudaDeviceSynchronize();
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel launch error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            delete[] bloom_filter_host;
            cudaFree(bloom_filter_device);
            return false;
        }
        
        cudaMemcpyFromSymbol(&found_flag, g_found, sizeof(int));
        if (found_flag) {
            printf("\n\n");
            
            char found_hex[65], found_hash160[41];
            cudaMemcpyFromSymbol(found_hex, g_found_hex, 65);
            cudaMemcpyFromSymbol(found_hash160, g_found_hash160, 41);
            
            double total_time = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start_time
            ).count();
            
            printf("========================\n");
            printf("🎉 MATCH FOUND! 🎉\n");
            printf("========================\n");
            printf("Private Key: %s\n", found_hex);
            printf("Hash160: %s\n", found_hash160);
            printf("------------------------\n");
            printf("Total time: %.2f seconds\n", total_time);
            printf("Total keys checked: %llu (%.2f billion)\n", 
                   (unsigned long long)total_keys_checked,
                   total_keys_checked / 1000000000.0);
            printf("Average speed: %.2f MK/s\n", total_keys_checked / total_time / 1000000.0);
            printf("========================\n");
            
            std::ofstream outfile("result.txt", std::ios::app);
            if (outfile.is_open()) {
                std::time_t now = std::time(nullptr);
                char timestamp[100];
                std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                outfile << "[" << timestamp << "] Found: " << found_hex << " -> " << found_hash160 << std::endl;
                outfile << "Total keys checked: " << total_keys_checked << std::endl;
                outfile << "Time taken: " << total_time << " seconds" << std::endl;
                outfile << "Average speed: " << (total_keys_checked / total_time / 1000000.0) << " MK/s" << std::endl;
                outfile << "Bloom filter: " << bloom_file << " (" << (bloom_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
                outfile << std::endl;
                outfile.close();
                std::cout << "Result appended to result.txt" << std::endl;
            }
            
            
            delete[] bloom_filter_host;
            cudaFree(bloom_filter_device);
            return true;
        }
        
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed_since_print = std::chrono::duration<double>(current_time - last_print_time).count();
        
        if (elapsed_since_print >= 1.0) {
            double total_time = std::chrono::duration<double>(current_time - start_time).count();
            double current_kps = total_keys_checked / total_time;
            
            printf("\rSpeed: %.2f MK/s | Total: %.2f B keys | Time: %.0fs",
                   current_kps / 1000000.0,
                   total_keys_checked / 1000000000.0,
                   total_time);
            fflush(stdout);
            
            last_print_time = current_time;
        }
    }
}

__global__ void debug_bloom_check(uint8_t* bloom_filter, uint64_t bloom_size_bytes, 
                                   const uint8_t* test_hash160, int* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== DEBUG BLOOM CHECK ===\n");
        printf("Bloom size: %llu bytes (%llu bits)\n", bloom_size_bytes, bloom_size_bytes * 8);
        
        printf("Test hash160 (bytes): ");
        for (int i = 0; i < 20; i++) {
            printf("%02x", test_hash160[i]);
        }
        printf("\n");
        
        uint64_t bloom_bits = bloom_size_bytes * 8;
        bool found = true;
        
        for (int i = 0; i < NUM_HASH_FUNCTIONS; i++) {
            uint64_t hash = bloom_size_bytes;  
            hash = i * 0x9e3779b1;
            
            
            for (int j = 0; j < 20; j++) {
                hash = hash * 0x100000001b3ULL ^ test_hash160[j];
            }
            
            uint64_t bit_index = hash % bloom_bits;
            uint64_t byte_index = bit_index / 8;
            uint8_t bit_offset = bit_index % 8;
            uint8_t byte_value = bloom_filter[byte_index];
            bool bit_set = (byte_value & (1 << bit_offset)) != 0;
            
            printf("Hash #%d: hash=%016llx, bit_idx=%llu, byte_idx=%llu, bit_off=%u, byte_val=%02x, bit_set=%d\n",
                   i, hash, bit_index, byte_index, bit_offset, byte_value, bit_set);
            
            if (!bit_set) {
                found = false;
            }
        }
        
        printf("Final result: %s\n", found ? "FOUND" : "NOT FOUND");
        printf("========================\n\n");
        
        *result = found ? 1 : 0;
    }
}


void test_bloom_with_known_address(const char* bloom_file, const char* test_hash160_hex) {
    printf("\n=== Testing Bloom Filter ===\n");
    printf("Bloom file: %s\n", bloom_file);
    printf("Test hash160: %s\n", test_hash160_hex);
    
    
    uint8_t *bloom_filter_host = nullptr;
    uint8_t *bloom_filter_device = nullptr;
    size_t bloom_size = 0;
    
    if (!load_bloom_filter(bloom_file, &bloom_filter_host, &bloom_filter_device, &bloom_size)) {
        return;
    }
    
    
    uint8_t test_hash160[20];
    hex_string_to_bytes(test_hash160_hex, test_hash160, 20);
    
    printf("Test hash160 bytes: ");
    for (int i = 0; i < 20; i++) {
        printf("%02x", test_hash160[i]);
    }
    printf("\n");
    
    
    uint8_t* d_test_hash160;
    cudaMalloc(&d_test_hash160, 20);
    cudaMemcpy(d_test_hash160, test_hash160, 20, cudaMemcpyHostToDevice);
    
    
    uint64_t bloom_size_u64 = bloom_size;
    cudaMemcpyToSymbol(d_bloom_size_bytes, &bloom_size_u64, sizeof(uint64_t));
    
    
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    
    
    debug_bloom_check<<<1, 1>>>(bloom_filter_device, bloom_size, d_test_hash160, d_result);
    cudaDeviceSynchronize();
    
    int result;
    cudaMemcpyFromSymbol(&result, d_result, sizeof(int));
    
    printf("\nCUDA Result: %s\n", result ? "✓ FOUND" : "✗ NOT FOUND");
    
    
    cudaFree(d_test_hash160);
    cudaFree(d_result);
    delete[] bloom_filter_host;
    cudaFree(bloom_filter_device);
    
    printf("===========================\n\n");
}

#define NUM_HASH_FUNCTIONS 10
#define BLOOM_SIZE_BYTES 1668452  


__global__ void test_bloom_hash(const uint8_t* hash160) {
    for (int i = 0; i < NUM_HASH_FUNCTIONS; i++) {
        uint64_t seed = (uint64_t)i * 0x9e3779b1ULL;
        uint64_t hv = fnv1a_hash_cuda(hash160, 20, seed);
        uint64_t bit_index = hv % (BLOOM_SIZE_BYTES * 8ULL);
        uint64_t byte_index = bit_index / 8;
        uint8_t bit_offset = bit_index % 8;

        printf("Hash#%d: seed=%016llx, hv=%016llx, bit_idx=%llu, byte_idx=%llu, bit_off=%u\n",
               i, seed, hv, bit_index, byte_index, bit_offset);
    }
}


void run_test() {
    uint8_t h_hash160[20] = {
        0x29,0xa7,0x82,0x13,0xca,0xa9,0xee,0xa8,0x24,0xac,
        0xf0,0x80,0x22,0xab,0x9d,0xfc,0x83,0x41,0x4f,0x56
    };

    uint8_t* d_hash160;
    cudaMalloc(&d_hash160, 20);
    cudaMemcpy(d_hash160, h_hash160, 20, cudaMemcpyHostToDevice);

    test_bloom_hash<<<1,1>>>(d_hash160);
    cudaDeviceSynchronize();

    cudaFree(d_hash160);
}


__global__ void verify_hash_calculation() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        
        uint8_t test_hash[20] = {
            0x62, 0xe9, 0x07, 0xb1, 0x5c, 0xbf, 0x27, 0xd5,
            0x42, 0x53, 0x99, 0xeb, 0xf6, 0xf0, 0xfb, 0x50,
            0xeb, 0xb8, 0x8f, 0x18
        };
        
        printf("\n=== CUDA HASH VERIFICATION ===\n");
        printf("Test hash160: ");
        for (int i = 0; i < 20; i++) {
            printf("%02x", test_hash[i]);
        }
        printf("\n\n");
        
        printf("Expected values from Python:\n");
        printf("Hash#0: seed=0000000000000000, hash=c69d7795927a565f, bit_idx=8017503\n");
        printf("Hash#1: seed=9e3779b97f4a7c13, hash=62c46fd9ff07d720, bit_idx=513824\n");
        printf("Hash#2: seed=3c6ef372fe94f826, hash=8fdc868359f42869, bit_idx=16001129\n");
        printf("Hash#9: seed=8ff34785799e5cab, hash=1eec8611e6e340b8, bit_idx=14893240\n");
        
        printf("\nCUDA calculated values:\n");
        
        uint64_t bloom_bits = 16777216ULL; 
        
        for (int i = 0; i < NUM_HASH_FUNCTIONS; i++) {
            uint64_t seed = (uint64_t)i * GOLDEN64;
            uint64_t hv = fnv1a_hash_cuda(test_hash, 20, seed);
            uint64_t bit_idx = hv % bloom_bits;
            uint64_t byte_idx = bit_idx / 8;
            uint8_t bit_off = bit_idx % 8;
            
            printf("Hash#%d: seed=%016llx, hash=%016llx, bit_idx=%llu, byte=%llu, bit_off=%u\n",
                   i, seed, hv, bit_idx, byte_idx, bit_off);
        }
        
        printf("\nIf all hash values match Python, the algorithm is correct!\n");
        printf("==============================\n\n");
    }
}


void test_hash_algorithm() {
    printf("\nTesting hash algorithm...\n");
    verify_hash_calculation<<<1, 1>>>();
    cudaDeviceSynchronize();
}


__global__ void debug_bloom_check_v2(
    uint8_t* bloom_filter, 
    uint64_t bloom_size_bytes,
    const uint8_t* test_hash160,
    int* result
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== CUDA BLOOM CHECK ===\n");
        printf("Bloom size: %llu bytes (%llu bits)\n", 
               bloom_size_bytes, bloom_size_bytes * 8ULL);
        
        printf("Hash160: ");
        for (int i = 0; i < 20; i++) {
            printf("%02x", test_hash160[i]);
        }
        printf("\n\n");
        
        uint64_t bloom_bits = bloom_size_bytes * 8ULL;
        bool found = true;
        
        for (int i = 0; i < NUM_HASH_FUNCTIONS; i++) {
            uint64_t seed = (uint64_t)i * GOLDEN64;
            uint64_t hv = fnv1a_hash_cuda(test_hash160, 20, seed);
            uint64_t bit_idx = hv % bloom_bits;
            uint64_t byte_idx = bit_idx / 8;
            uint8_t bit_off = bit_idx % 8;
            uint8_t byte_val = bloom_filter[byte_idx];
            
            
            bool bit_set = (byte_val & (1 << (7 - bit_off))) != 0;
            
            printf("Hash#%d: seed=%016llx, hash=%016llx, bit_idx=%llu, "
                   "byte=%llu, bit_off=%u, byte_val=%02x, bit_set=%d\n",
                   i, seed, hv, bit_idx, byte_idx, bit_off, byte_val, bit_set);
            
            if (!bit_set) {
                found = false;
            }
        }
        
        printf("\nResult: %s\n", found ? "✓ FOUND" : "✗ NOT FOUND");
        printf("========================\n\n");
        
        *result = found ? 1 : 0;
    }
}


void test_bloom_filter_cuda(const char* bloom_file, const char* hash160_hex) {
    printf("\n=== Testing Bloom Filter with CUDA ===\n");
    printf("Bloom file: %s\n", bloom_file);
    printf("Hash160: %s\n\n", hash160_hex);
    
    
    test_hash_algorithm();
    
    
    uint8_t *bloom_host = nullptr;
    uint8_t *bloom_device = nullptr;
    size_t bloom_size = 0;
    
    if (!load_bloom_filter(bloom_file, &bloom_host, &bloom_device, &bloom_size)) {
        return;
    }
    
    
    uint8_t hash160[20];
    hex_string_to_bytes(hash160_hex, hash160, 20);
    
    printf("Testing hash160 bytes: ");
    for (int i = 0; i < 20; i++) {
        printf("%02x", hash160[i]);
    }
    printf("\n");
    
    
    uint8_t* d_hash160;
    int* d_result;
    cudaMalloc(&d_hash160, 20);
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_hash160, hash160, 20, cudaMemcpyHostToDevice);
    
    
    debug_bloom_check_v2<<<1, 1>>>(bloom_device, bloom_size, d_hash160, d_result);
    cudaDeviceSynchronize();
    
    
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\nFinal CUDA result: %s\n", result ? "✓ FOUND" : "✗ NOT FOUND");
    
    
    cudaFree(d_hash160);
    cudaFree(d_result);
    delete[] bloom_host;
    cudaFree(bloom_device);
    
    printf("=====================================\n");
}

int main(int argc, char* argv[]) {
	
	
	if (argc >= 3 && strcmp(argv[1], "test") == 0) {
		if (argc < 4) {
			std::cerr << "Usage: " << argv[0] << " test <bloom_file> <hash160_hex>" << std::endl;
			return 1;
		}
		
		init_gpu_constants();
		cudaDeviceSynchronize();
		
		test_bloom_filter_cuda(argv[2], argv[3]);
		return 0;
	}
    
	
    
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <min> <max> <bloom_filter_file> [device_id]" << std::endl;
        std::cerr << "       " << argv[0] << " test <bloom_filter_file> <hash160_hex>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 20000000000000000 3ffffffffffffffff bloom.bin" << std::endl;
        return 1;
    }
    
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int blocks = prop.multiProcessorCount * 16;
    int threads = 256;
    int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;
    
    cudaSetDevice(device_id);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error setting device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    
    
   
   
    
    init_gpu_constants();
    cudaDeviceSynchronize();
    bool result = run_with_bloom_filter(argv[1], argv[2], argv[3], blocks, threads, device_id);

    return 0;
}