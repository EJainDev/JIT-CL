#ifndef JIT_CL_JIT_H
#define JIT_CL_JIT_H

#include <CL/opencl.hpp>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "jit/ops.h"
#include "jit/state.h"
#include "jit/tracer.h"

namespace jitcl {

inline auto genKernel(cl::Context context, const JitState state,
                      const std::vector<JitTracer>& params) -> cl::Kernel {
  std::string kernel = "__kernel void kern(";
  for (const auto& tracer : params) {
    // Parameter
    kernel += std::format(
        "__global float *{0}, __global int *{0}_shape, __global int *{0}_stride, int {0}_dims",
        tracer._data->name);
  }
  kernel = kernel.substr(0, kernel.size() - 2);  // Remove the last ", "
  kernel += R"() {
const int gid0 = get_global_id(0);
const int gid1 = get_global_id(1);
const int global0 = get_global_size(0);
const int global1 = get_global_size(1);
)";
  for (const auto& tracer : params) {
    kernel += std::format("if (gid0 >= {0}_shape[0] || gid1 >= {0}_shape[1]) return;\n",
                          tracer._data->name);
  }
  for (const auto& operation : state.get()->operations) {
    switch (operation.op) {
      case internal::Operations::None:
        break;
      case internal::Operations::Assign: {
        kernel += std::format("{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {0}_stride[gid0]];\n",
                              operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Add: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] + {2}[gid1 * "
            "{2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Subtract: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] - {2}[gid1 * "
            "{2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Multiply: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] * {2}[gid1 * "
            "{2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Divide: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] / {2}[gid1 * "
            "{2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::And: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = (int){1}[gid1 * {1}_stride[gid0]] & (int){2}[gid1 * "
            "{2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Or: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = (int){1}[gid1 * {1}_stride[gid0]] | (int){2}[gid1 * "
            "{2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Xor: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = (int){1}[gid1 * {1}_stride[gid0]] ^ (int){2}[gid1 * "
            "{2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Reduction: {
        kernel += std::format(R"(float reduce_var = 0.0;
for (int i = global1 - 1; i > 0; i >>= 1) {{
  if (gid1 < i) {{
    {0}[gid1 * {0}_stride[gid0]] += {0}[gid1 * {0}_stride[gid0]] + i];
  }}
}})",
                              operation.lhs._data->name);
      }
      case internal::Operations::AssignScalar: {
        kernel += std::format("{0}[gid1 * {0}_stride[gid0]] = {1};\n", operation.lhs._data->name,
                              operation.rhs_scalar);
        break;
      }
      case internal::Operations::AddScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {0}_stride[gid0]] + {2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::SubtractScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {0}_stride[gid0]] - {2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::MultiplyScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {0}_stride[gid0]] * {2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::DivideScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {0}_stride[gid0]] / {2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::AndScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = (int){1}[gid1 * {0}_stride[gid0]] & (int){2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::OrScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = (int){1}[gid1 * {0}_stride[gid0]] | (int){2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::XorScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = (int){1}[gid1 * {0}_stride[gid0]] ^ (int){2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      
      // Comparison operations
      case internal::Operations::Equal: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] == {2}[gid1 * {2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::NotEqual: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] != {2}[gid1 * {2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::LessThan: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] < {2}[gid1 * {2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::LessThanOrEqual: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] <= {2}[gid1 * {2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::GreaterThan: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] > {2}[gid1 * {2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::GreaterThanOrEqual: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] >= {2}[gid1 * {2}_stride[gid0]];\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::EqualScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] == {2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::NotEqualScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] != {2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::LessThanScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] < {2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::LessThanOrEqualScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] <= {2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::GreaterThanScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] > {2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::GreaterThanOrEqualScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = {1}[gid1 * {1}_stride[gid0]] >= {2};\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      
      // Trigonometric functions
      case internal::Operations::Sin: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = sin({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Cos: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = cos({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Tan: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = tan({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Asin: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = asin({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Acos: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = acos({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Atan: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = atan({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::SinScalar:
      case internal::Operations::CosScalar:
      case internal::Operations::TanScalar:
      case internal::Operations::AsinScalar:
      case internal::Operations::AcosScalar:
      case internal::Operations::AtanScalar:
        // Scalar trig functions don't make sense in this context - ignore
        break;
        
      // Hyperbolic functions
      case internal::Operations::Sinh: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = sinh({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Cosh: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = cosh({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Tanh: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = tanh({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Asinh: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = asinh({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Acosh: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = acosh({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Atanh: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = atanh({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::SinhScalar:
      case internal::Operations::CoshScalar:
      case internal::Operations::TanhScalar:
      case internal::Operations::AsinhScalar:
      case internal::Operations::AcoshScalar:
      case internal::Operations::AtanhScalar:
        // Scalar hyperbolic functions don't make sense - ignore
        break;
        
      // Exponential and logarithmic functions
      case internal::Operations::Exp: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = exp({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Exp2: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = exp2({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Exp10: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = exp10({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Expm1: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = expm1({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Log: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = log({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Log2: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = log2({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Log10: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = log10({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Log1p: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = log1p({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::ExpScalar:
      case internal::Operations::Exp2Scalar:
      case internal::Operations::Exp10Scalar:
      case internal::Operations::Expm1Scalar:
      case internal::Operations::LogScalar:
      case internal::Operations::Log2Scalar:
      case internal::Operations::Log10Scalar:
      case internal::Operations::Log1pScalar:
        // Scalar exp/log functions don't make sense - ignore
        break;
        
      // Power functions
      case internal::Operations::Pow: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = pow({1}[gid1 * {1}_stride[gid0]], {2}[gid1 * {2}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Pown: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = pown({1}[gid1 * {1}_stride[gid0]], (int){2}[gid1 * {2}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Powr: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = powr({1}[gid1 * {1}_stride[gid0]], {2}[gid1 * {2}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Sqrt: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = sqrt({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::PowScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = pow({1}[gid1 * {1}_stride[gid0]], {2});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::PownScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = pown({1}[gid1 * {1}_stride[gid0]], (int){2});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::PowrScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = powr({1}[gid1 * {1}_stride[gid0]], {2});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::SqrtScalar:
        // Scalar sqrt doesn't make sense - ignore
        break;
        
      // Rounding and absolute value functions
      case internal::Operations::Floor: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = floor({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Ceil: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = ceil({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Round: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = round({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Trunc: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = trunc({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Abs: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fabs({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Fabs: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fabs({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Sign: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = sign({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Fract: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fract({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::FloorScalar:
      case internal::Operations::CeilScalar:
      case internal::Operations::RoundScalar:
      case internal::Operations::TruncScalar:
      case internal::Operations::AbsScalar:
      case internal::Operations::FabsScalar:
      case internal::Operations::SignScalar:
      case internal::Operations::FractScalar:
        // Scalar rounding functions don't make sense - ignore
        break;
        
      // Min/Max/Clamp functions
      case internal::Operations::Min: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fmin({1}[gid1 * {1}_stride[gid0]], {2}[gid1 * {2}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Max: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fmax({1}[gid1 * {1}_stride[gid0]], {2}[gid1 * {2}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Clamp: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = clamp({1}[gid1 * {1}_stride[gid0]], {2}[gid1 * {2}_stride[gid0]], {3}[gid1 * {3}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name, operation.rhs2._data->name);
        break;
      }
      case internal::Operations::Fmod: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fmod({1}[gid1 * {1}_stride[gid0]], {2}[gid1 * {2}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::MinScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fmin({1}[gid1 * {1}_stride[gid0]], {2});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::MaxScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fmax({1}[gid1 * {1}_stride[gid0]], {2});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::ClampScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = clamp({1}[gid1 * {1}_stride[gid0]], {2}, {3});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar, operation.rhs2_scalar);
        break;
      }
      case internal::Operations::FmodScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fmod({1}[gid1 * {1}_stride[gid0]], {2});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      
      // Additional useful functions
      case internal::Operations::Copysign: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = copysign({1}[gid1 * {1}_stride[gid0]], {2}[gid1 * {2}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Fma: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fma({1}[gid1 * {1}_stride[gid0]], {2}[gid1 * {2}_stride[gid0]], {3}[gid1 * {3}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name, operation.rhs2._data->name);
        break;
      }
      case internal::Operations::Mad: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = mad({1}[gid1 * {1}_stride[gid0]], {2}[gid1 * {2}_stride[gid0]], {3}[gid1 * {3}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name, operation.rhs2._data->name);
        break;
      }
      case internal::Operations::Rsqrt: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = rsqrt({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::CopysignScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = copysign({1}[gid1 * {1}_stride[gid0]], {2});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::FmaScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = fma({1}[gid1 * {1}_stride[gid0]], {2}, {3});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar, operation.rhs2_scalar);
        break;
      }
      case internal::Operations::MadScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = mad({1}[gid1 * {1}_stride[gid0]], {2}, {3});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar, operation.rhs2_scalar);
        break;
      }
      case internal::Operations::RsqrtScalar:
        // Scalar rsqrt doesn't make sense - ignore
        break;
        
      // Special mathematical functions
      case internal::Operations::Cbrt: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = cbrt({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Hypot: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = hypot({1}[gid1 * {1}_stride[gid0]], {2}[gid1 * {2}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Erf: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = erf({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::Erfc: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = erfc({1}[gid1 * {1}_stride[gid0]]);\n",
            operation.output._data->name, operation.lhs._data->name);
        break;
      }
      case internal::Operations::HypotScalar: {
        kernel += std::format(
            "{0}[gid1 * {0}_stride[gid0]] = hypot({1}[gid1 * {1}_stride[gid0]], {2});\n",
            operation.output._data->name, operation.lhs._data->name, operation.rhs_scalar);
        break;
      }
      case internal::Operations::CbrtScalar:
      case internal::Operations::ErfScalar:
      case internal::Operations::ErfcScalar:
        // Scalar special functions don't make sense - ignore
        break;
    }
  }

  kernel += "}\n";

  std::cout << kernel << '\n';

  cl::Program program(context, kernel);
  program.build();
  return {program, "kern"};
}
}  // namespace jitcl

#endif  // JIT_CL_JIT_H
