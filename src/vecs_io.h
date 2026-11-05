// Readers for the TEXMEX .fvecs / .ivecs formats.
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Both formats store each record as a 4-byte dimension followed by d 4-byte values.
template <typename T>
std::vector<T> read_vecs(const std::string& path, int64_t& n, int64_t& d) {
    static_assert(sizeof(T) == 4, "vecs files store 4-byte values");
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const int64_t bytes = f.tellg();
    f.seekg(0);

    int32_t dim = 0;
    f.read(reinterpret_cast<char*>(&dim), 4);
    if (!f || dim <= 0) throw std::runtime_error("bad header in " + path);
    d = dim;
    const int64_t record = 4 + d * 4;
    if (bytes % record != 0) throw std::runtime_error("bad file size for " + path);
    n = bytes / record;

    std::vector<T> data(n * d);
    f.seekg(0);
    for (int64_t i = 0; i < n; i++) {
        int32_t row_dim = 0;
        f.read(reinterpret_cast<char*>(&row_dim), 4);
        if (row_dim != dim) throw std::runtime_error("inconsistent dimension in " + path);
        f.read(reinterpret_cast<char*>(&data[i * d]), d * 4);
    }
    if (!f) throw std::runtime_error("short read from " + path);
    return data;
}
