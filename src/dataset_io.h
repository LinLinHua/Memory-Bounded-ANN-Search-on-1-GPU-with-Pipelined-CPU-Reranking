// Loading and saving vector files.
//   .fvecs / .ivecs : TEXMEX format (SIFT, GIST), read with vecs_io.h
//   .fbin  / .ibin  : big-ann-benchmarks format (DEEP), header = int32 n, int32 d
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vecs_io.h"

inline bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Reads the first max_rows rows of a .fbin/.ibin file (all rows when max_rows < 0).
template <typename T>
std::vector<T> read_bin(const std::string& path, int64_t& n, int64_t& d, int64_t max_rows = -1) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    int32_t header[2] = {0, 0};
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!f || header[0] <= 0 || header[1] <= 0) throw std::runtime_error("bad header in " + path);
    n = header[0];
    d = header[1];
    if (max_rows >= 0 && max_rows < n) n = max_rows;

    std::vector<T> data(n * d);
    f.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(T));
    if (!f) throw std::runtime_error("short read from " + path);
    return data;
}

// Chooses the reader from the file extension.
template <typename T>
std::vector<T> load_vectors(const std::string& path, int64_t& n, int64_t& d, int64_t max_rows = -1) {
    if (ends_with(path, "vecs")) {
        auto data = read_vecs<T>(path, n, d);
        if (max_rows >= 0 && max_rows < n) {
            n = max_rows;
            data.resize(n * d);
        }
        return data;
    }
    if (ends_with(path, "bin")) return read_bin<T>(path, n, d, max_rows);
    throw std::runtime_error("unknown file type: " + path);
}

// Writes n rows of d int32 values as .ivecs (the ground-truth format).
inline void write_ivecs(const std::string& path, const int32_t* data, int64_t n, int64_t d) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    const int32_t dim = static_cast<int32_t>(d);
    for (int64_t i = 0; i < n; i++) {
        f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        f.write(reinterpret_cast<const char*>(data + i * d), d * sizeof(int32_t));
    }
    if (!f) throw std::runtime_error("write failed for " + path);
}
