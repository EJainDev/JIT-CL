#ifndef JIT_CL_OPS_H_
#define JIT_CL_OPS_H_

#include <cstdint>

#include "tracer.h"

namespace jitcl::internal {
enum class Operations : std::uint16_t {
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
  AssignScalar,
  AddScalar,
  SubtractScalar,
  MultiplyScalar,
  DivideScalar,
  AndScalar,
  OrScalar,
  XorScalar,
  
  // Comparison operations
  Equal,
  NotEqual,
  LessThan,
  LessThanOrEqual,
  GreaterThan,
  GreaterThanOrEqual,
  EqualScalar,
  NotEqualScalar,
  LessThanScalar,
  LessThanOrEqualScalar,
  GreaterThanScalar,
  GreaterThanOrEqualScalar,
  
  // Trigonometric functions
  Sin,
  Cos,
  Tan,
  Asin,
  Acos,
  Atan,
  SinScalar,
  CosScalar,
  TanScalar,
  AsinScalar,
  AcosScalar,
  AtanScalar,
  
  // Hyperbolic functions
  Sinh,
  Cosh,
  Tanh,
  Asinh,
  Acosh,
  Atanh,
  SinhScalar,
  CoshScalar,
  TanhScalar,
  AsinhScalar,
  AcoshScalar,
  AtanhScalar,
  
  // Exponential and logarithmic functions
  Exp,
  Exp2,
  Exp10,
  Expm1,
  Log,
  Log2,
  Log10,
  Log1p,
  ExpScalar,
  Exp2Scalar,
  Exp10Scalar,
  Expm1Scalar,
  LogScalar,
  Log2Scalar,
  Log10Scalar,
  Log1pScalar,
  
  // Power functions
  Pow,
  Pown,
  Powr,
  Sqrt,
  PowScalar,
  PownScalar,
  PowrScalar,
  SqrtScalar,
  
  // Rounding and absolute value functions
  Floor,
  Ceil,
  Round,
  Trunc,
  Abs,
  Fabs,
  Sign,
  Fract,
  FloorScalar,
  CeilScalar,
  RoundScalar,
  TruncScalar,
  AbsScalar,
  FabsScalar,
  SignScalar,
  FractScalar,
  
  // Min/Max/Clamp functions
  Min,
  Max,
  Clamp,
  Fmod,
  MinScalar,
  MaxScalar,
  ClampScalar,
  FmodScalar,
  
  // Additional useful functions
  Copysign,
  Fma,
  Mad,
  Rsqrt,
  CopysignScalar,
  FmaScalar,
  MadScalar,
  RsqrtScalar,
  
  // Special mathematical functions
  Cbrt,
  Hypot,
  Erf,
  Erfc,
  CbrtScalar,
  HypotScalar,
  ErfScalar,
  ErfcScalar,
};

struct OperationStack {
  Operations op{Operations::None};
  JitTracer output{nullptr};
  JitTracer lhs{nullptr};
  JitTracer rhs{nullptr};
  JitTracer rhs2{nullptr};  // For ternary operations like clamp, fma
  float rhs_scalar{0.0f};
  float rhs2_scalar{0.0f};  // For scalar ternary operations
};
}  // namespace jitcl::internal

#endif