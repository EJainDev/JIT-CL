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
