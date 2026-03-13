#include <CL/opencl.hpp>
#include <memory>

#include "jit.h"
#include "jit/state.h"
#include "jit/tracer.h"

using namespace jitcl;

void copy(JitTracer a, JitTracer b) { a.set(b); }

JitTracer add(JitTracer a, JitTracer b) { return a.add(b); }

void reduce(JitTracer a) { a.reduce(); }

auto main() -> int {
  auto state = JitState(std::make_shared<internal::JitState>());
  JitTracer a{state};
  JitTracer b{state};
  copy(a, b);
  auto c = add(a, b);
  reduce(c);

  genKernel(cl::Context(), state, {a, b, c});

  return 0;
}
