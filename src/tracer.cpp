#include "jit/tracer.h"

#include <memory>
#include <string>

#include "jit/ops.h"
#include "jit/state.h"

namespace jitcl {
JitTracer::JitTracer(JitState state) : _data(std::make_shared<Data>()) {
  _data->state = state;
  _data->name = "t" + std::to_string(state->tracer_count++);
}

void JitTracer::set(JitTracer other) const {
  if (other._data->name == _data->name) {
    return;
  }
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::Assign,
                                      .output = JitTracer(nullptr),
                                      .lhs = *this,
                                      .rhs = other});
  _data->state->mtx.unlock();
}

auto JitTracer::add(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Add, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::sub(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Subtract, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::mul(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Multiply, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::div(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Divide, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::bitwise_and(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::And, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::bitwise_or(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Or, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::bitwise_xor(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Xor, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}
void JitTracer::reduce() const {
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Reduction, .lhs = *this, .rhs = JitTracer(nullptr)});
  _data->state->mtx.unlock();
}

JitTracer JitTracer::add(float other) const {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::AddScalar, .output = result, .lhs = *this, .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

JitTracer JitTracer::sub(float other) const {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::SubtractScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

JitTracer JitTracer::mul(float other) const {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::MultiplyScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

JitTracer JitTracer::div(float other) const {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::DivideScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

JitTracer JitTracer::bitwise_and(float other) const {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::AndScalar, .output = result, .lhs = *this, .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

JitTracer JitTracer::bitwise_or(float other) const {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::OrScalar, .output = result, .lhs = *this, .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

JitTracer JitTracer::bitwise_xor(float other) const {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::XorScalar, .output = result, .lhs = *this, .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

// Comparison operations
auto JitTracer::equal(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Equal, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::not_equal(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::NotEqual, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::less_than(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::LessThan, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::less_than_or_equal(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::LessThanOrEqual, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::greater_than(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::GreaterThan, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::greater_than_or_equal(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::GreaterThanOrEqual,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::equal(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::EqualScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::not_equal(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::NotEqualScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::less_than(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::LessThanScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::less_than_or_equal(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::LessThanOrEqualScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::greater_than(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::GreaterThanScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::greater_than_or_equal(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::GreaterThanOrEqualScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

// Trigonometric functions
auto JitTracer::sin() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Sin, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::cos() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Cos, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::tan() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Tan, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::asin() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Asin, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::acos() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Acos, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::atan() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Atan, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

// Hyperbolic functions
auto JitTracer::sinh() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Sinh, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::cosh() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Cosh, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::tanh() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Tanh, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::asinh() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Asinh, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::acosh() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Acosh, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::atanh() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Atanh, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

// Exponential and logarithmic functions
auto JitTracer::exp() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Exp, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::exp2() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Exp2, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::exp10() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Exp10, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::expm1() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Expm1, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::log() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Log, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::log2() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Log2, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::log10() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Log10, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::log1p() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Log1p, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

// Power functions
auto JitTracer::pow(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Pow, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::pown(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Pown, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::powr(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Powr, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::sqrt() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Sqrt, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::pow(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::PowScalar, .output = result, .lhs = *this, .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::pown(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::PownScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::powr(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::PowrScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

// Rounding and absolute value functions
auto JitTracer::floor() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Floor, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::ceil() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Ceil, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::round() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Round, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::trunc() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Trunc, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::abs() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Abs, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::fabs() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Fabs, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::sign() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Sign, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::fract() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Fract, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

// Min/Max/Clamp functions
auto JitTracer::min(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Min, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::max(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Max, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::clamp(JitTracer min_val, JitTracer max_val) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::Clamp,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs = min_val,
                                      .rhs2 = max_val});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::fmod(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Fmod, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::min(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::MinScalar, .output = result, .lhs = *this, .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::max(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::MaxScalar, .output = result, .lhs = *this, .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::clamp(float min_val, float max_val) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::ClampScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = min_val,
                                      .rhs2_scalar = max_val});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::fmod(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::FmodScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

// Additional useful functions
auto JitTracer::copysign(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Copysign, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::rsqrt() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Rsqrt, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::copysign(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::CopysignScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}

// Special mathematical functions
auto JitTracer::cbrt() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Cbrt, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::hypot(JitTracer other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Hypot, .output = result, .lhs = *this, .rhs = other});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::erf() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Erf, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::erfc() const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back(
      {.op = internal::Operations::Erfc, .output = result, .lhs = *this});
  _data->state->mtx.unlock();
  return result;
}

auto JitTracer::hypot(float other) const -> JitTracer {
  JitTracer result(_data->state);
  _data->state->mtx.lock();
  _data->state->operations.push_back({.op = internal::Operations::HypotScalar,
                                      .output = result,
                                      .lhs = *this,
                                      .rhs_scalar = other});
  _data->state->mtx.unlock();
  return result;
}
}  // namespace jitcl