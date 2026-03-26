#include <CL/opencl.hpp>
#include <iostream>
#include <memory>

#include "jit.h"
#include "jit/state.h"
#include "jit/tracer.h"

using namespace jitcl;

void test_basic_ops(JitState state) {
  std::cout << "\n=== Testing Basic Operations ===\n";
  JitTracer a{state};
  JitTracer b{state};
  auto c = a.add(b);
  auto d = a.sub(b);
  auto e = a.mul(b);
  auto f = a.div(b);
}

void test_comparison_ops(JitState state) {
  std::cout << "\n=== Testing Comparison Operations ===\n";
  JitTracer a{state};
  JitTracer b{state};
  auto eq = a.equal(b);
  auto ne = a.not_equal(b);
  auto lt = a.less_than(b);
  auto le = a.less_than_or_equal(b);
  auto gt = a.greater_than(b);
  auto ge = a.greater_than_or_equal(b);
  auto eq_scalar = a.equal(1.0f);
  auto lt_scalar = a.less_than(5.0f);
}

void test_trig_ops(JitState state) {
  std::cout << "\n=== Testing Trigonometric Operations ===\n";
  JitTracer a{state};
  auto s = a.sin();
  auto c = a.cos();
  auto t = a.tan();
  auto as = a.asin();
  auto ac = a.acos();
  auto at = a.atan();
}

void test_hyperbolic_ops(JitState state) {
  std::cout << "\n=== Testing Hyperbolic Operations ===\n";
  JitTracer a{state};
  auto sh = a.sinh();
  auto ch = a.cosh();
  auto th = a.tanh();
  auto ash = a.asinh();
  auto ach = a.acosh();
  auto ath = a.atanh();
}

void test_explog_ops(JitState state) {
  std::cout << "\n=== Testing Exponential/Logarithmic Operations ===\n";
  JitTracer a{state};
  auto e = a.exp();
  auto e2 = a.exp2();
  auto e10 = a.exp10();
  auto em1 = a.expm1();
  auto l = a.log();
  auto l2 = a.log2();
  auto l10 = a.log10();
  auto l1p = a.log1p();
}

void test_power_ops(JitState state) {
  std::cout << "\n=== Testing Power Operations ===\n";
  JitTracer a{state};
  JitTracer b{state};
  auto p = a.pow(b);
  auto pn = a.pown(b);
  auto pr = a.powr(b);
  auto sq = a.sqrt();
  auto p_scalar = a.pow(2.0f);
  auto pn_scalar = a.pown(3.0f);
}

void test_rounding_ops(JitState state) {
  std::cout << "\n=== Testing Rounding Operations ===\n";
  JitTracer a{state};
  auto fl = a.floor();
  auto ce = a.ceil();
  auto ro = a.round();
  auto tr = a.trunc();
  auto ab = a.abs();
  auto fa = a.fabs();
  auto si = a.sign();
  auto fr = a.fract();
}

void test_minmax_ops(JitState state) {
  std::cout << "\n=== Testing Min/Max/Clamp Operations ===\n";
  JitTracer a{state};
  JitTracer b{state};
  JitTracer c{state};
  auto min_val = a.min(b);
  auto max_val = a.max(b);
  auto clamp_val = a.clamp(b, c);
  auto fmod_val = a.fmod(b);
  auto min_scalar = a.min(0.0f);
  auto max_scalar = a.max(1.0f);
  auto clamp_scalar = a.clamp(0.0f, 1.0f);
}

void test_additional_ops(JitState state) {
  std::cout << "\n=== Testing Additional Operations ===\n";
  JitTracer a{state};
  JitTracer b{state};
  JitTracer c{state};
  auto cs = a.copysign(b);
  auto fm = a.fma(b, c);
  auto md = a.mad(b, c);
  auto rs = a.rsqrt();
  auto cs_scalar = a.copysign(1.0f);
}

void test_special_ops(JitState state) {
  std::cout << "\n=== Testing Special Operations ===\n";
  JitTracer a{state};
  JitTracer b{state};
  auto cb = a.cbrt();
  auto hy = a.hypot(b);
  auto er = a.erf();
  auto erc = a.erfc();
  auto hy_scalar = a.hypot(3.0f);
}

void test_chained_ops(JitState state) {
  std::cout << "\n=== Testing Chained Operations ===\n";
  JitTracer a{state};
  JitTracer b{state};
  // Test: (sin(a) + cos(b)) * exp(a.pow(2))
  auto s = a.sin();
  auto c = b.cos();
  auto sum = s.add(c);
  auto a_sq = a.pow(2.0f);
  auto e = a_sq.exp();
  auto result = sum.mul(e);
}

void test_scalar_operations(JitState state) {
  std::cout << "\n=== Testing Scalar Operations ===\n";
  JitTracer a{state};
  auto add_s = a.add(5.0f);
  auto mul_s = a.mul(2.0f);
  auto pow_s = a.pow(2.0f);
  auto clamp_s = a.clamp(-1.0f, 1.0f);
}

auto main() -> int {
  std::cout << "JIT-CL Comprehensive Test Suite\n";
  std::cout << "================================\n";

  auto state = JitState(std::make_shared<internal::JitState>());

  // Run all tests
  test_basic_ops(state);
  test_comparison_ops(state);
  test_trig_ops(state);
  test_hyperbolic_ops(state);
  test_explog_ops(state);
  test_power_ops(state);
  test_rounding_ops(state);
  test_minmax_ops(state);
  test_additional_ops(state);
  test_special_ops(state);
  test_chained_ops(state);
  test_scalar_operations(state);

  // Generate kernel to verify all operations compile correctly
  std::cout << "\n=== Generated Kernel Code ===\n";
  JitTracer input1{state};
  JitTracer input2{state};

  try {
    genKernel(cl::Context(), state, {input1, input2});
    std::cout << "\n=== All Operations Successfully Compiled! ===\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
