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
  std::string kernel = R"(
    __kernel void kernel(
  )";
  for (const auto& tracer : params) {
    // Parameter
    kernel += std::format("__global float *{}, ", tracer._data->name);
  }
  kernel = kernel.substr(0, kernel.size() - 2);  // Remove the last ", "
  kernel += ") {\nconst int gid0 = get_global_id(0);\nconst int global0 = get_global_size(0);\n";
  for (const auto& operation : state.get()->operations) {
    switch (operation.op) {
      case internal::Operations::None:
        break;
      case internal::Operations::Assign: {
        kernel += std::format("{}[gid0] = {}[gid0];\n", operation.lhs._data->name,
                              operation.rhs._data->name);
        break;
      }
      case internal::Operations::Add: {
        kernel += std::format("{}[gid0] = {}[gid0] + {}[gid0];\n", operation.output._data->name,
                              operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Subtract: {
        kernel += std::format("{}[gid0] = {}[gid0] - {}[gid0];\n", operation.output._data->name,
                              operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Multiply: {
        kernel += std::format("{}[gid0] = {}[gid0] * {}[gid0];\n", operation.output._data->name,
                              operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Divide: {
        kernel += std::format("{}[gid0] = {}[gid0] / {}[gid0];\n", operation.output._data->name,
                              operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::And: {
        kernel +=
            std::format("{}[gid0] = (int){}[gid0] & (int){}[gid0];\n", operation.output._data->name,
                        operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Or: {
        kernel +=
            std::format("{}[gid0] = (int){}[gid0] | (int){}[gid0];\n", operation.output._data->name,
                        operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Xor: {
        kernel +=
            std::format("{}[gid0] = (int){}[gid0] ^ (int){}[gid0];\n", operation.output._data->name,
                        operation.lhs._data->name, operation.rhs._data->name);
        break;
      }
      case internal::Operations::Reduction: {
        kernel += std::format(R"(float reduce_var = 0.0;
for (int i = global0 - 1; i > 0; i >>= 1) {{
if (gid0 < i) {{
{0}[gid0] += {0}[gid0 + i];
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
  return {program, "kernel"};
}
}  // namespace jitcl

#endif  // JIT_CL_JIT_H
