// The IVF index written by ivf_build (same layout as build_ivf.py), loaded into host memory.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct IvfIndex {
    int64_t n = 0, d = 0, nlist = 0;
    std::vector<float> centroids;   // nlist x d
    std::vector<int64_t> offsets;   // nlist + 1; list l owns rows offsets[l] .. offsets[l+1]-1
    std::vector<int32_t> ids;       // n; original vector ID of each row
    std::vector<uint8_t> codes8;    // n x d
    std::vector<uint8_t> codes4;    // n x d/2; low nibble = even dimension
    std::vector<uint32_t> codes1;   // n x d/32; sign bits of the rotated residual
    std::vector<float> norms;       // n; |residual|
    std::vector<float> sq_min;      // d; residual = sq_min + code * step
    std::vector<float> sq_step8;    // d
    std::vector<float> sq_step4;    // d
    std::vector<float> rotation;    // d x d

    int64_t list_size(int64_t l) const { return offsets[l + 1] - offsets[l]; }
};

namespace ivf_detail {

template <typename T>
std::vector<T> read_raw(const std::string& path, int64_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    if (static_cast<int64_t>(f.tellg()) != count * static_cast<int64_t>(sizeof(T)))
        throw std::runtime_error("unexpected size of " + path);
    std::vector<T> data(count);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    return data;
}

// Reads an integer field such as  "nlist": 1024  from meta.json.
inline int64_t json_int(const std::string& text, const std::string& key) {
    const auto at = text.find("\"" + key + "\"");
    if (at == std::string::npos) throw std::runtime_error("meta.json has no " + key);
    return std::atoll(text.c_str() + text.find(':', at) + 1);
}

}  // namespace ivf_detail

inline IvfIndex load_ivf(const std::string& dir) {
    using ivf_detail::read_raw;
    std::ifstream meta_file(dir + "/meta.json");
    if (!meta_file) throw std::runtime_error("cannot open " + dir + "/meta.json");
    std::stringstream meta;
    meta << meta_file.rdbuf();

    IvfIndex ix;
    ix.n = ivf_detail::json_int(meta.str(), "n");
    ix.d = ivf_detail::json_int(meta.str(), "d");
    ix.nlist = ivf_detail::json_int(meta.str(), "nlist");
    const int64_t n = ix.n, d = ix.d, L = ix.nlist;

    ix.centroids = read_raw<float>(dir + "/centroids.f32", L * d);
    ix.offsets = read_raw<int64_t>(dir + "/list_offsets.i64", L + 1);
    ix.ids = read_raw<int32_t>(dir + "/ids.i32", n);
    ix.codes8 = read_raw<uint8_t>(dir + "/codes8.u8", n * d);
    ix.codes4 = read_raw<uint8_t>(dir + "/codes4.u8", n * d / 2);
    ix.codes1 = read_raw<uint32_t>(dir + "/codes1.u32", n * d / 32);
    ix.norms = read_raw<float>(dir + "/norms.f32", n);
    ix.sq_min = read_raw<float>(dir + "/sq_min.f32", d);
    ix.sq_step8 = read_raw<float>(dir + "/sq_step8.f32", d);
    ix.sq_step4 = read_raw<float>(dir + "/sq_step4.f32", d);
    ix.rotation = read_raw<float>(dir + "/rotation.f32", d * d);
    if (ix.offsets.front() != 0 || ix.offsets.back() != n)
        throw std::runtime_error("list offsets do not cover all vectors in " + dir);
    return ix;
}
