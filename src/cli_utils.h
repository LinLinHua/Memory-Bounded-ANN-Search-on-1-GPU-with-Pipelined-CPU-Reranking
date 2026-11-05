// Small helpers shared by LL's programs: "--key value" options, CSV output, medians.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Command-line options of the form --key value.
class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            const std::string key = argv[i];
            if (key.rfind("--", 0) != 0 || i + 1 >= argc)
                throw std::runtime_error("expected --key value, got " + key);
            values_[key.substr(2)] = argv[++i];
        }
    }

    std::string str(const std::string& key, const std::string& fallback = "") const {
        used_.insert(key);
        const auto it = values_.find(key);
        return it == values_.end() ? fallback : it->second;
    }

    std::string required(const std::string& key) const {
        if (!values_.count(key)) throw std::runtime_error("--" + key + " is required");
        return str(key);
    }

    int64_t num(const std::string& key, int64_t fallback) const {
        const std::string s = str(key);
        return s.empty() ? fallback : std::atoll(s.c_str());
    }

    // "16,32,64" -> {16, 32, 64}
    std::vector<int64_t> list(const std::string& key, const std::string& fallback) const {
        std::vector<int64_t> out;
        std::stringstream ss(str(key, fallback));
        std::string item;
        while (std::getline(ss, item, ',')) out.push_back(std::atoll(item.c_str()));
        return out;
    }

    // Call after reading all options, so a typo fails instead of being ignored.
    void reject_unknown() const {
        for (const auto& kv : values_)
            if (!used_.count(kv.first)) throw std::runtime_error("unknown option --" + kv.first);
    }

private:
    std::map<std::string, std::string> values_;
    mutable std::set<std::string> used_;
};

// Opens a CSV for appending; writes the header only when the file is new or empty.
inline FILE* open_csv(const std::string& path, const char* header) {
    if (path.empty()) return nullptr;
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    if (std::ftell(f) == 0) std::fprintf(f, "%s\n", header);
    return f;
}

inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
