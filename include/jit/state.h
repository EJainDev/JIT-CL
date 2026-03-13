#ifndef JIT_CL_STATE_H
#define JIT_CL_STATE_H

#include <memory>
#include <mutex>
#include <vector>

namespace jitcl {
namespace internal {
struct OperationStack;

struct JitState {
  std::mutex mtx;
  int tracer_count{0};
  std::vector<OperationStack> operations;
};
}  // namespace internal

using JitState = std::shared_ptr<internal::JitState>;
}  // namespace jitcl

#endif  // JIT_CL_STATE_H
