#ifndef JIT_CL_TRACER_H
#define JIT_CL_TRACER_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "state.h"

namespace jitcl {
class JitTracer {
 public:
  explicit JitTracer(JitState state);

  explicit JitTracer(std::nullptr_t) : _data(nullptr) {};

  void set(JitTracer other) const;
  [[nodiscard]] auto add(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto sub(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto mul(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto div(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto bitwise_and(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto bitwise_or(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto bitwise_xor(JitTracer other) const -> JitTracer;
  void reduce() const;

  struct Data {
    std::string name;
    std::vector<int> shape;
    std::vector<int> stride;
    JitState state;
  };

  std::shared_ptr<Data> _data;
};
}  // namespace jitcl

#endif  // JIT_CL_TRACER_H
