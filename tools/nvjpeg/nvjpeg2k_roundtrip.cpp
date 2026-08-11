#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

#include <chrono>
#include <cstdint>
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

void check_nvjpeg2k(nvjpeg2kStatus_t status, const char *what) {
    if (status != NVJPEG2K_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": nvjpeg2k status " + std::to_string(status));
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
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: nvjpeg2k_roundtrip <input_u16.raw> <width> <height> [depth] <target_psnr> <output_u16.raw>\n";
        return 2;
    }

    try {
        const std::string input_path = argv[1];
        const uint32_t width = static_cast<uint32_t>(std::atoi(argv[2]));
        const uint32_t height = static_cast<uint32_t>(std::atoi(argv[3]));
        const uint32_t depth = argc == 7 ? static_cast<uint32_t>(std::atoi(argv[4])) : 1;
        const double target_psnr = std::atof(argv[argc == 7 ? 5 : 4]);
        const std::string output_path = argv[argc == 7 ? 6 : 5];
        if (width == 0 || height == 0 || depth == 0 || target_psnr <= 0.0) {
            throw std::runtime_error("invalid width, height, depth, or target_psnr");
        }

        auto input_bytes = read_file(input_path);
        const size_t elements_per_slice = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t slice_bytes = elements_per_slice * sizeof(uint16_t);
        const size_t expected_bytes = slice_bytes * static_cast<size_t>(depth);
        if (input_bytes.size() != expected_bytes) {
            throw std::runtime_error("input size does not match width*height*depth*sizeof(uint16)");
        }
        std::vector<unsigned char> output_bytes(expected_bytes);

        cudaStream_t stream{};
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

        uint16_t *d_input = nullptr;
        uint16_t *d_output = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_input), slice_bytes), "cudaMalloc input");
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_output), slice_bytes), "cudaMalloc output");

        void *input_planes[1] = {d_input};
        size_t input_pitch[1] = {static_cast<size_t>(width) * sizeof(uint16_t)};
        nvjpeg2kImage_t input_image{};
        input_image.pixel_data = input_planes;
        input_image.pitch_in_bytes = input_pitch;
        input_image.pixel_type = NVJPEG2K_UINT16;
        input_image.num_components = 1;

        nvjpeg2kEncoder_t encoder{};
        nvjpeg2kEncodeParams_t encode_params{};
        check_nvjpeg2k(nvjpeg2kEncoderCreateSimple(&encoder), "nvjpeg2kEncoderCreateSimple");
        check_nvjpeg2k(nvjpeg2kEncodeParamsCreate(&encode_params), "nvjpeg2kEncodeParamsCreate");

        nvjpeg2kImageComponentInfo_t comp{};
        comp.component_width = width;
        comp.component_height = height;
        comp.precision = 16;
        comp.sgn = 0;

        nvjpeg2kEncodeConfig_t config{};
        config.stream_type = NVJPEG2K_STREAM_J2K;
        config.color_space = NVJPEG2K_COLORSPACE_GRAY;
        config.rsiz = 0;
        config.image_width = width;
        config.image_height = height;
        config.enable_tiling = 0;
        config.tile_width = width;
        config.tile_height = height;
        config.num_components = 1;
        config.image_comp_info = &comp;
        config.enable_SOP_marker = 0;
        config.enable_EPH_marker = 0;
        config.prog_order = NVJPEG2K_LRCP;
        config.num_layers = 1;
        config.mct_mode = 0;
        config.num_resolutions = 6;
        config.code_block_w = 64;
        config.code_block_h = 64;
        config.encode_modes = 0;
        config.irreversible = 1;
        config.num_precincts_init = 0;

        check_nvjpeg2k(nvjpeg2kEncodeParamsSetEncodeConfig(encode_params, &config), "nvjpeg2kEncodeParamsSetEncodeConfig");
        check_nvjpeg2k(nvjpeg2kEncodeParamsSetQuality(encode_params, target_psnr), "nvjpeg2kEncodeParamsSetQuality");
        check_nvjpeg2k(nvjpeg2kEncodeParamsSetInputFormat(encode_params, NVJPEG2K_FORMAT_PLANAR), "nvjpeg2kEncodeParamsSetInputFormat");

        nvjpeg2kHandle_t handle{};
        check_nvjpeg2k(nvjpeg2kCreateSimple(&handle), "nvjpeg2kCreateSimple");

        void *output_planes[1] = {d_output};
        size_t output_pitch[1] = {static_cast<size_t>(width) * sizeof(uint16_t)};
        nvjpeg2kImage_t output_image{};
        output_image.pixel_data = output_planes;
        output_image.pitch_in_bytes = output_pitch;
        output_image.pixel_type = NVJPEG2K_UINT16;
        output_image.num_components = 1;

        size_t total_codestream_size = 0;
        long long encode_us_total = 0;
        long long decode_us_total = 0;

        for (uint32_t z = 0; z < depth; ++z) {
            const size_t offset = static_cast<size_t>(z) * slice_bytes;
            check_cuda(cudaMemcpyAsync(d_input, input_bytes.data() + offset, slice_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");

            nvjpeg2kEncodeState_t encode_state{};
            check_nvjpeg2k(nvjpeg2kEncodeStateCreate(encoder, &encode_state), "nvjpeg2kEncodeStateCreate");
            auto t0 = std::chrono::high_resolution_clock::now();
            check_nvjpeg2k(nvjpeg2kEncode(encoder, encode_state, encode_params, &input_image, stream), "nvjpeg2kEncode");
            check_cuda(cudaStreamSynchronize(stream), "encode synchronize");
            encode_us_total += micros_since(t0);

            size_t codestream_size = 0;
            check_nvjpeg2k(nvjpeg2kEncodeRetrieveBitstream(encoder, encode_state, nullptr, &codestream_size, stream), "nvjpeg2kEncodeRetrieveBitstream size");
            std::vector<unsigned char> codestream(codestream_size);
            check_nvjpeg2k(nvjpeg2kEncodeRetrieveBitstream(encoder, encode_state, codestream.data(), &codestream_size, stream), "nvjpeg2kEncodeRetrieveBitstream data");
            codestream.resize(codestream_size);
            total_codestream_size += codestream.size();
            nvjpeg2kEncodeStateDestroy(encode_state);

            nvjpeg2kDecodeState_t decode_state{};
            nvjpeg2kStream_t jpeg2k_stream{};
            check_nvjpeg2k(nvjpeg2kDecodeStateCreate(handle, &decode_state), "nvjpeg2kDecodeStateCreate");
            check_nvjpeg2k(nvjpeg2kStreamCreate(&jpeg2k_stream), "nvjpeg2kStreamCreate");
            check_nvjpeg2k(nvjpeg2kStreamParse(handle, codestream.data(), codestream.size(), 0, 0, jpeg2k_stream), "nvjpeg2kStreamParse");

            t0 = std::chrono::high_resolution_clock::now();
            check_nvjpeg2k(nvjpeg2kDecode(handle, decode_state, jpeg2k_stream, &output_image, stream), "nvjpeg2kDecode");
            check_cuda(cudaStreamSynchronize(stream), "decode synchronize");
            decode_us_total += micros_since(t0);

            check_cuda(cudaMemcpyAsync(output_bytes.data() + offset, d_output, slice_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
            check_cuda(cudaStreamSynchronize(stream), "copy synchronize");
            nvjpeg2kStreamDestroy(jpeg2k_stream);
            nvjpeg2kDecodeStateDestroy(decode_state);
        }

        write_file(output_path, output_bytes);

        std::cout << "{\"j2k_bytes\":" << total_codestream_size
                  << ",\"encode_us\":" << encode_us_total
                  << ",\"decode_us\":" << decode_us_total
                  << ",\"width\":" << width
                  << ",\"height\":" << height
                  << ",\"depth\":" << depth
                  << ",\"target_psnr\":" << target_psnr
                  << "}\n";

        nvjpeg2kDestroy(handle);
        nvjpeg2kEncodeParamsDestroy(encode_params);
        nvjpeg2kEncoderDestroy(encoder);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception &exc) {
        std::cerr << "nvjpeg2k_roundtrip error: " << exc.what() << "\n";
        return 1;
    }
}
