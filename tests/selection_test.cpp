#include "../src/selection.h"
#include <cassert>
#include <cstdio>

int main() {
  const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
  std::vector<Candidate> candidates = {
    {"strict", "fp32", "p0", 2.0, 0.0004},
    {"relaxed", "fp32", "p1", 1.0, 0.003},
    {"bf16", "bf16", "p2", 0.5, 0.006},
    {"nan_error", "fp32", "p3", 0.01, nan},
    {"inf_error", "fp32", "p4", 0.01, inf},
    {"nan_time", "fp32", "p5", nan, 0},
    {"zero_time", "fp32", "p6", 0, 0},
    {"negative_error", "fp32", "p7", 0.01, -1}
  };
  assert(select_candidate(candidates, 0.001, "fp32") == &candidates[0]);
  assert(select_candidate(candidates, 0.01, "fp32") == &candidates[1]);
  assert(select_candidate(candidates, 0.01, "bf16") == &candidates[2]);
  assert(select_candidate(candidates, 0.01, "any") == &candidates[2]);
  assert(select_candidate(candidates, 0.0004, "fp32") == &candidates[0]);
  assert(!select_candidate(candidates, 0.0001, "fp32"));
  assert(!select_candidate(candidates, 0.001, "bf16"));
  assert(!select_candidate(candidates, nan, "fp32"));
  assert(!select_candidate(candidates, inf, "fp32"));
  assert(!select_candidate(candidates, -1, "fp32"));
  assert(!select_candidate(candidates, 0, "fp32"));
  assert(!select_candidate(candidates, 0.01, "unknown"));
  assert(!select_candidate({}, 0.01, "fp32"));
  candidates.push_back({"tie", "fp32", "p8", 1.0, 0.002});
  assert(select_candidate(candidates, 0.01, "fp32") == &candidates.back());
  assert(relative_l2(0, 0) == 0);
  assert(std::isinf(relative_l2(1, 0)));
  assert(relative_l2(1, 100) == 0.1);
  std::puts("PASS: error target, I/O isolation, invalid candidates, ties, and zero reference");
}
