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

  void set(float other) const;
  [[nodiscard]] auto add(float other) const -> JitTracer;
  [[nodiscard]] auto sub(float other) const -> JitTracer;
  [[nodiscard]] auto mul(float other) const -> JitTracer;
  [[nodiscard]] auto div(float other) const -> JitTracer;
  [[nodiscard]] auto bitwise_and(float other) const -> JitTracer;
  [[nodiscard]] auto bitwise_or(float other) const -> JitTracer;
  [[nodiscard]] auto bitwise_xor(float other) const -> JitTracer;

  // Comparison operations
  [[nodiscard]] auto equal(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto not_equal(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto less_than(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto less_than_or_equal(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto greater_than(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto greater_than_or_equal(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto equal(float other) const -> JitTracer;
  [[nodiscard]] auto not_equal(float other) const -> JitTracer;
  [[nodiscard]] auto less_than(float other) const -> JitTracer;
  [[nodiscard]] auto less_than_or_equal(float other) const -> JitTracer;
  [[nodiscard]] auto greater_than(float other) const -> JitTracer;
  [[nodiscard]] auto greater_than_or_equal(float other) const -> JitTracer;

  // Trigonometric functions
  [[nodiscard]] auto sin() const -> JitTracer;
  [[nodiscard]] auto cos() const -> JitTracer;
  [[nodiscard]] auto tan() const -> JitTracer;
  [[nodiscard]] auto asin() const -> JitTracer;
  [[nodiscard]] auto acos() const -> JitTracer;
  [[nodiscard]] auto atan() const -> JitTracer;

  // Hyperbolic functions
  [[nodiscard]] auto sinh() const -> JitTracer;
  [[nodiscard]] auto cosh() const -> JitTracer;
  [[nodiscard]] auto tanh() const -> JitTracer;
  [[nodiscard]] auto asinh() const -> JitTracer;
  [[nodiscard]] auto acosh() const -> JitTracer;
  [[nodiscard]] auto atanh() const -> JitTracer;

  // Exponential and logarithmic functions
  [[nodiscard]] auto exp() const -> JitTracer;
  [[nodiscard]] auto exp2() const -> JitTracer;
  [[nodiscard]] auto exp10() const -> JitTracer;
  [[nodiscard]] auto expm1() const -> JitTracer;
  [[nodiscard]] auto log() const -> JitTracer;
  [[nodiscard]] auto log2() const -> JitTracer;
  [[nodiscard]] auto log10() const -> JitTracer;
  [[nodiscard]] auto log1p() const -> JitTracer;

  // Power functions
  [[nodiscard]] auto pow(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto pown(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto powr(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto sqrt() const -> JitTracer;
  [[nodiscard]] auto pow(float other) const -> JitTracer;
  [[nodiscard]] auto pown(float other) const -> JitTracer;
  [[nodiscard]] auto powr(float other) const -> JitTracer;

  // Rounding and absolute value functions
  [[nodiscard]] auto floor() const -> JitTracer;
  [[nodiscard]] auto ceil() const -> JitTracer;
  [[nodiscard]] auto round() const -> JitTracer;
  [[nodiscard]] auto trunc() const -> JitTracer;
  [[nodiscard]] auto abs() const -> JitTracer;
  [[nodiscard]] auto fabs() const -> JitTracer;
  [[nodiscard]] auto sign() const -> JitTracer;
  [[nodiscard]] auto fract() const -> JitTracer;

  // Min/Max/Clamp functions
  [[nodiscard]] auto min(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto max(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto clamp(JitTracer min_val, JitTracer max_val) const -> JitTracer;
  [[nodiscard]] auto fmod(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto min(float other) const -> JitTracer;
  [[nodiscard]] auto max(float other) const -> JitTracer;
  [[nodiscard]] auto clamp(float min_val, float max_val) const -> JitTracer;
  [[nodiscard]] auto fmod(float other) const -> JitTracer;

  // Additional useful functions
  [[nodiscard]] auto copysign(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto fma(JitTracer mul_val, JitTracer add_val) const -> JitTracer;
  [[nodiscard]] auto mad(JitTracer mul_val, JitTracer add_val) const -> JitTracer;
  [[nodiscard]] auto rsqrt() const -> JitTracer;
  [[nodiscard]] auto copysign(float other) const -> JitTracer;

  // Special mathematical functions
  [[nodiscard]] auto cbrt() const -> JitTracer;
  [[nodiscard]] auto hypot(JitTracer other) const -> JitTracer;
  [[nodiscard]] auto erf() const -> JitTracer;
  [[nodiscard]] auto erfc() const -> JitTracer;
  [[nodiscard]] auto hypot(float other) const -> JitTracer;

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
