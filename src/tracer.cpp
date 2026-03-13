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
  _data->state->operations.push_back({.op = internal::Operations::Reduction,
                                      .output = JitTracer(nullptr),
                                      .lhs = *this,
                                      .rhs = JitTracer(nullptr)});
  _data->state->mtx.unlock();
}
}  // namespace jitcl