#pragma once
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

struct Candidate {
  std::string method, io, config;
  double ms, rel_err;
};

inline const Candidate* select_candidate(const std::vector<Candidate>& candidates, double max_error, const char* io) {
  if (!std::isfinite(max_error) || max_error <= 0) return nullptr;
  if (std::strcmp(io, "fp32") && std::strcmp(io, "bf16") && std::strcmp(io, "any")) return nullptr;
  const Candidate* best = nullptr;
  for (const auto& candidate : candidates) {
    if (std::strcmp(io, "any") && std::strcmp(io, candidate.io.c_str())) continue;
    if (!std::isfinite(candidate.ms) || candidate.ms <= 0 || !std::isfinite(candidate.rel_err) ||
        candidate.rel_err < 0 || candidate.rel_err > max_error) continue;
    if (!best || candidate.ms < best->ms || (candidate.ms == best->ms && candidate.rel_err < best->rel_err)) best = &candidate;
  }
  return best;
}

inline double relative_l2(double squared_error, double squared_reference) {
  if (squared_reference > 0) return std::sqrt(squared_error / squared_reference);
  return squared_error == 0 ? 0 : std::numeric_limits<double>::infinity();
}
