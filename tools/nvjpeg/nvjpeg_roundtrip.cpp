#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_nvjpeg(nvjpegStatus_t status, const char *what) {
    if (status != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": nvjpeg status " + std::to_string(status));
    }
}

std::vector<unsigned char> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input: " + path);
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<unsigned char> data(static_cast<size_t>(size));
    in.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}

void write_file(const std::string &path, const std::vector<unsigned char> &data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path);
    }
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
}

long long micros_since(std::chrono::high_resolution_clock::time_point start) {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "usage: nvjpeg_roundtrip <input_rgb_u8.raw> <width> <height> <quality> <output_rgb_u8.raw>\n";
        return 2;
    }

    try {
        const std::string input_path = argv[1];
        const int width = std::atoi(argv[2]);
        const int height = std::atoi(argv[3]);
        const int quality = std::atoi(argv[4]);
        const std::string output_path = argv[5];
        if (width <= 0 || height <= 0 || quality < 1 || quality > 100) {
            throw std::runtime_error("invalid width, height, or quality");
        }

        auto input = read_file(input_path);
        const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
        if (input.size() != expected) {
            throw std::runtime_error("input size does not match width*height*3");
        }

        cudaStream_t stream{};
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

        nvjpegHandle_t handle{};
        nvjpegJpegState_t decode_state{};
        nvjpegEncoderState_t encode_state{};
        nvjpegEncoderParams_t encode_params{};

        check_nvjpeg(nvjpegCreateSimple(&handle), "nvjpegCreateSimple");
        check_nvjpeg(nvjpegJpegStateCreate(handle, &decode_state), "nvjpegJpegStateCreate");
        check_nvjpeg(nvjpegEncoderStateCreate(handle, &encode_state, stream), "nvjpegEncoderStateCreate");
        check_nvjpeg(nvjpegEncoderParamsCreate(handle, &encode_params, stream), "nvjpegEncoderParamsCreate");
        check_nvjpeg(nvjpegEncoderParamsSetQuality(encode_params, quality, stream), "nvjpegEncoderParamsSetQuality");
        check_nvjpeg(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, 1, stream), "nvjpegEncoderParamsSetOptimizedHuffman");
        check_nvjpeg(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_444, stream), "nvjpegEncoderParamsSetSamplingFactors");

        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_input), expected), "cudaMalloc input");
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_output), expected), "cudaMalloc output");
        check_cuda(cudaMemcpyAsync(d_input, input.data(), expected, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");

        nvjpegImage_t source{};
        source.channel[0] = d_input;
        source.pitch[0] = static_cast<unsigned int>(width * 3);

        auto t0 = std::chrono::high_resolution_clock::now();
        check_nvjpeg(nvjpegEncodeImage(handle, encode_state, encode_params, &source, NVJPEG_INPUT_RGBI, width, height, stream), "nvjpegEncodeImage");
        check_cuda(cudaStreamSynchronize(stream), "encode synchronize");
        const long long encode_us = micros_since(t0);

        size_t jpeg_size = 0;
        check_nvjpeg(nvjpegEncodeRetrieveBitstream(handle, encode_state, nullptr, &jpeg_size, stream), "nvjpegEncodeRetrieveBitstream size");
        std::vector<unsigned char> jpeg(jpeg_size);
        check_nvjpeg(nvjpegEncodeRetrieveBitstream(handle, encode_state, jpeg.data(), &jpeg_size, stream), "nvjpegEncodeRetrieveBitstream data");
        jpeg.resize(jpeg_size);

        nvjpegImage_t destination{};
        destination.channel[0] = d_output;
        destination.pitch[0] = static_cast<unsigned int>(width * 3);

        t0 = std::chrono::high_resolution_clock::now();
        check_nvjpeg(nvjpegDecode(handle, decode_state, jpeg.data(), jpeg.size(), NVJPEG_OUTPUT_RGBI, &destination, stream), "nvjpegDecode");
        check_cuda(cudaStreamSynchronize(stream), "decode synchronize");
        const long long decode_us = micros_since(t0);

        std::vector<unsigned char> output(expected);
        check_cuda(cudaMemcpyAsync(output.data(), d_output, expected, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
        check_cuda(cudaStreamSynchronize(stream), "copy synchronize");
        write_file(output_path, output);

        std::cout << "{\"jpeg_bytes\":" << jpeg.size()
                  << ",\"encode_us\":" << encode_us
                  << ",\"decode_us\":" << decode_us
                  << ",\"width\":" << width
                  << ",\"height\":" << height
                  << ",\"quality\":" << quality
                  << "}\n";

        cudaFree(d_input);
        cudaFree(d_output);
        nvjpegEncoderParamsDestroy(encode_params);
        nvjpegEncoderStateDestroy(encode_state);
        nvjpegJpegStateDestroy(decode_state);
        nvjpegDestroy(handle);
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception &exc) {
        std::cerr << "nvjpeg_roundtrip error: " << exc.what() << "\n";
        return 1;
    }
}
