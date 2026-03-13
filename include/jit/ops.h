#ifndef JIT_CL_OPS_H_
#define JIT_CL_OPS_H_

#include <cstdint>

#include "tracer.h"

namespace jitcl::internal {
enum class Operations : std::uint8_t {
  None,
  Assign,
  Add,
  Subtract,
  Multiply,
  Divide,
  And,
  Or,
  Xor,
  Reduction,
};

struct OperationStack {
  Operations op{Operations::None};
  JitTracer output{nullptr};
  JitTracer lhs{nullptr};
  JitTracer rhs{nullptr};
};
}  // namespace jitcl::internal

#endif