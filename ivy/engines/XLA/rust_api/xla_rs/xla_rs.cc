#include "xla_rs.h"

#define ASSIGN_OR_RETURN_STATUS(lhs, rexpr)                                    \
  ASSIGN_OR_RETURN_STATUS_IMPL(                                                \
      TF_STATUS_MACROS_CONCAT_NAME(_statusor, __COUNTER__), lhs, rexpr)

#define ASSIGN_OR_RETURN_STATUS_IMPL(statusor, lhs, rexpr)                     \
  auto statusor = (rexpr);                                                     \
  if (!statusor.ok())                                                          \
    return new Status(statusor.status());                                      \
  auto lhs = std::move(statusor.value());

#define MAYBE_RETURN_STATUS(rexpr)                                             \
  MAYBE_RETURN_STATUS_IMPL(TF_STATUS_MACROS_CONCAT_NAME(_status, __COUNTER__), \
                           rexpr)

#define MAYBE_RETURN_STATUS_IMPL(statusor, rexpr)                              \
  auto statusor = (rexpr);                                                     \
  if (!statusor.ok())                                                          \
    return new Status(statusor);

#define BEGIN_PROTECT_OP try {
#define END_PROTECT_OP_B(builder)                                              \
  }                                                                            \
  catch (std::exception e) {                                                   \
    return new XlaOp(builder->ReportError(tsl::errors::Internal(e.what())));   \
  }
#define END_PROTECT_OP(arg)                                                    \
  }                                                                            \
  catch (std::exception e) {                                                   \
    return new XlaOp(                                                          \
        arg->builder()->ReportError(tsl::errors::Internal(e.what())));         \
  }

status pjrt_cpu_client_create(pjrt_client *output) {
  ASSIGN_OR_RETURN_STATUS(client, xla::GetTfrtCpuClient(false));
  *output = new std::shared_ptr(std::move(client));
  return nullptr;
}

status pjrt_gpu_client_create(pjrt_client *output, double memory_fraction,
                              bool preallocate) {
  xla::GpuAllocatorConfig allocator = {.memory_fraction = memory_fraction,
                                       .preallocate = preallocate};
  ASSIGN_OR_RETURN_STATUS(
      client, xla::GetStreamExecutorGpuClient(false, allocator, nullptr, 0));
  *output = new std::shared_ptr(std::move(client));
  return nullptr;
}

status pjrt_tpu_client_create(pjrt_client *output,
                              int max_inflight_computations) {
  ASSIGN_OR_RETURN_STATUS(client, xla::GetTpuClient(max_inflight_computations));
  *output = new std::shared_ptr(std::move(client));
  return nullptr;
}

int pjrt_client_device_count(pjrt_client c) { return (*c)->device_count(); }

int pjrt_client_addressable_device_count(pjrt_client c) {
  return (*c)->addressable_device_count();
}

void pjrt_client_devices(pjrt_client c, pjrt_device *outputs) {
  size_t index = 0;
  for (auto device : (*c)->devices()) {
    outputs[index++] = device;
  }
}

void pjrt_client_addressable_devices(pjrt_client c, pjrt_device *outputs) {
  size_t index = 0;
  for (auto device : (*c)->addressable_devices()) {
    outputs[index++] = device;
  }
}

char *pjrt_client_platform_name(pjrt_client c) {
  // TODO: Avoid the double allocation when converting string views.
  return strdup(std::string((*c)->platform_name()).c_str());
}

char *pjrt_client_platform_version(pjrt_client c) {
  return strdup(std::string((*c)->platform_version()).c_str());
}

void pjrt_client_free(pjrt_client b) { delete b; }

void pjrt_loaded_executable_free(pjrt_loaded_executable b) { delete b; }

status pjrt_buffer_from_host_buffer(const pjrt_client client,
                                    const pjrt_device device, const void *d,
                                    int pr_type, int dsize, const int64_t *ds,
                                    pjrt_buffer *output) {
  PjRtDevice *device_ = device == nullptr ? (*client)->devices()[0] : device;
  ASSIGN_OR_RETURN_STATUS(
      buffer,
      (*client)->BufferFromHostBuffer(
          d, (PrimitiveType)pr_type, absl::Span<const int64_t>(ds, dsize), {},
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, []() {},
          device_));
  *output = buffer.release();
  return nullptr;
}

status pjrt_buffer_from_host_literal(const pjrt_client client,
                                     const pjrt_device device, const literal l,
                                     pjrt_buffer *output) {
  PjRtDevice *d = device == nullptr ? (*client)->devices()[0] : device;
  ASSIGN_OR_RETURN_STATUS(buffer, (*client)->BufferFromHostLiteral(*l, d));
  *output = buffer.release();
  return nullptr;
}

status pjrt_buffer_to_literal_sync(pjrt_buffer b, literal *output) {
  ASSIGN_OR_RETURN_STATUS(literal, b->ToLiteralSync());
  *output = new Literal();
  **output = std::move(*literal);
  return nullptr;
}

shape pjrt_buffer_on_device_shape(pjrt_buffer b) {
  return new Shape(b->on_device_shape());
}

status pjrt_buffer_copy_to_device(pjrt_buffer b, pjrt_device device,
                                  pjrt_buffer *output) {
  ASSIGN_OR_RETURN_STATUS(copied_b, b->CopyToDevice(device));
  *output = copied_b.release();
  return nullptr;
}

status pjrt_buffer_copy_raw_to_host_sync(pjrt_buffer b, void *dst,
                                         size_t offset, size_t transfer_size) {
  MAYBE_RETURN_STATUS(b->CopyRawToHost(dst, offset, transfer_size).Await());
  return nullptr;
}

void pjrt_buffer_free(pjrt_buffer b) { delete b; }

int pjrt_device_id(pjrt_device d) { return d->id(); }

int pjrt_device_process_index(pjrt_device d) { return d->process_index(); }

int pjrt_device_local_hardware_id(pjrt_device d) {
  return d->local_hardware_id();
}

status pjrt_device_transfer_to_infeed(pjrt_device d, const literal l) {
  MAYBE_RETURN_STATUS(d->TransferToInfeed(*l));
  return nullptr;
}

status pjrt_device_transfer_from_outfeed(pjrt_device d, literal l) {
  MAYBE_RETURN_STATUS(d->TransferFromOutfeed(l));
  return nullptr;
}

char *pjrt_device_kind(pjrt_device d) {
  return strdup(std::string(d->device_kind()).c_str());
}

char *pjrt_device_debug_string(pjrt_device d) {
  return strdup(std::string(d->DebugString()).c_str());
}

char *pjrt_device_to_string(pjrt_device d) {
  return strdup(std::string(d->ToString()).c_str());
}

xla_builder xla_builder_create(const char *name) {
  return new XlaBuilder(name);
}

void xla_builder_free(xla_builder b) { delete b; }

xla_op constant_literal(const xla_builder b, const literal l) {
  BEGIN_PROTECT_OP
  return new XlaOp(ConstantLiteral(b, *l));
  END_PROTECT_OP_B(b)
}

#define CONST_OP_R01(native_type, primitive_type)                              \
  xla_op constant_r0_##native_type(const xla_builder b, native_type f) {       \
    return new XlaOp(ConstantR0<native_type>(b, f));                           \
  }                                                                            \
  xla_op constant_r1c_##native_type(const xla_builder b, native_type f,        \
                                    size_t len) {                              \
    return new XlaOp(ConstantR1<native_type>(b, len, f));                      \
  }                                                                            \
  xla_op constant_r1_##native_type(const xla_builder b, const native_type *f,  \
                                   size_t len) {                               \
    return new XlaOp(                                                          \
        ConstantR1<native_type>(b, absl::Span<const native_type>(f, len)));    \
  }                                                                            \
  literal create_r0_##native_type(native_type f) {                             \
    return new Literal(LiteralUtil::CreateR0<native_type>(f));                 \
  }                                                                            \
  literal create_r1_##native_type(const native_type *f, size_t nel) {          \
    return new Literal(LiteralUtil::CreateR1<native_type>(                     \
        absl::Span<const native_type>(f, nel)));                               \
  }                                                                            \
  native_type literal_get_first_element_##native_type(const literal l) {       \
    return l->GetFirstElement<native_type>();                                  \
  }

FOR_EACH_NATIVE_TYPE(CONST_OP_R01)
#undef CONST_OP_R01

Shape make_shape_internal(int pr_type, int dsize, const int64_t *ds) {
  bool has_negative_dim = false;
  for (int i = 0; i < dsize; ++i) {
    if (ds[i] < 0) {
      has_negative_dim = true;
      break;
    }
  }
  Shape shape;
  if (has_negative_dim) {
    std::vector<bool> dynamic;
    std::vector<int64_t> bounds;
    for (int i = 0; i < dsize; ++i) {
      if (ds[i] < 0) {
        bounds.push_back(-ds[i]);
        dynamic.push_back(true);
      } else {
        bounds.push_back(ds[i]);
        dynamic.push_back(false);
      }
    }
    shape = ShapeUtil::MakeShape(
        (PrimitiveType)pr_type,
        absl::Span<const int64_t>(bounds.data(), bounds.size()), dynamic);
  } else {
    shape = ShapeUtil::MakeShape((PrimitiveType)pr_type,
                                 absl::Span<const int64_t>(ds, dsize));
  }
  return shape;
}

shape make_shape_array(int pr_type, size_t dsize, const int64_t *ds) {
  return new Shape(make_shape_internal(pr_type, dsize, ds));
}

shape make_shape_tuple(size_t dsize, const shape *ds) {
  std::vector<Shape> elts;
  for (size_t i = 0; i < dsize; ++i) {
    elts.push_back(*ds[i]);
  }
  return new Shape(ShapeUtil::MakeTupleShape(elts));
}

xla_op parameter(const xla_builder b, int64_t id, int pr_type, int dsize,
                 const int64_t *ds, const char *name) {
  BEGIN_PROTECT_OP
  Shape shape = make_shape_internal(pr_type, dsize, ds);
  return new XlaOp(Parameter(b, id, shape, std::string(name)));
  END_PROTECT_OP_B(b)
}

xla_op parameter_s(const xla_builder b, int64_t id, const shape s,
                   const char *name) {
  BEGIN_PROTECT_OP
  return new XlaOp(Parameter(b, id, *s, std::string(name)));
  END_PROTECT_OP_B(b)
}

xla_op infeed(const xla_builder b, int pr_type, int dsize, const int64_t *ds,
              const char *config) {
  BEGIN_PROTECT_OP
  Shape shape = make_shape_internal(pr_type, dsize, ds);
  return new XlaOp(Infeed(b, shape, std::string(config)));
  END_PROTECT_OP_B(b)
}

void outfeed(const xla_op op, int pr_type, int dsize, const int64_t *ds,
             const char *outfeed_config) {
  Shape shape = make_shape_internal(pr_type, dsize, ds);
  Outfeed(*op, shape, std::string(outfeed_config));
}

xla_op op_add(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Add(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_sub(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Sub(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_mul(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Mul(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_div(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Div(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_rem(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Rem(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_max(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Max(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_min(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Min(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_and(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(And(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_or(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Or(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_xor(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Xor(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_atan2(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Atan2(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_pow(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Pow(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_dot(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Dot(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_dot_general(const xla_op lhs, const xla_op rhs, const int64_t *lhs_c,
                      size_t nlhs_c, const int64_t *rhs_c, size_t nrhs_c,
                      const int64_t *lhs_b, size_t nlhs_b, const int64_t *rhs_b,
                      size_t nrhs_b) {
  BEGIN_PROTECT_OP
  DotDimensionNumbers dnums;
  for (size_t i = 0; i < nlhs_c; ++i)
    dnums.add_lhs_contracting_dimensions(lhs_c[i]);
  for (size_t i = 0; i < nrhs_c; ++i)
    dnums.add_rhs_contracting_dimensions(rhs_c[i]);
  for (size_t i = 0; i < nlhs_b; ++i)
    dnums.add_lhs_batch_dimensions(lhs_b[i]);
  for (size_t i = 0; i < nrhs_b; ++i)
    dnums.add_rhs_batch_dimensions(rhs_b[i]);
  return new XlaOp(DotGeneral(*lhs, *rhs, dnums));
  END_PROTECT_OP(lhs)
}

xla_op op_eq(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Eq(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_ne(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Ne(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_ge(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Ge(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_gt(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Gt(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_le(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Le(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_lt(const xla_op lhs, const xla_op rhs) {
  BEGIN_PROTECT_OP
  return new XlaOp(Lt(*lhs, *rhs));
  END_PROTECT_OP(lhs)
}

xla_op op_shift_left(const xla_op lhs, const xla_op rhs) {
    BEGIN_PROTECT_OP
    return new XlaOp(ShiftLeft(*lhs, *rhs));
    END_PROTECT_OP(lhs)
}

xla_op op_shift_right_arith(const xla_op lhs, const xla_op rhs) {
    BEGIN_PROTECT_OP
    return new XlaOp(ShiftRightArithmetic(*lhs, *rhs));
    END_PROTECT_OP(lhs)
}

xla_op op_shift_right_logic(const xla_op lhs, const xla_op rhs) {
    BEGIN_PROTECT_OP
    return new XlaOp(ShiftRightLogical(*lhs, *rhs));
    END_PROTECT_OP(lhs)
}

xla_op op_population_count(const xla_op arg) {
    BEGIN_PROTECT_OP
    return new XlaOp(PopulationCount(*arg));
    END_PROTECT_OP(arg)
}

xla_op op_not(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Not(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_abs(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Abs(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_exp(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Exp(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_expm1(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Expm1(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_floor(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Floor(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_ceil(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Ceil(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_round(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Round(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_round_nearest_even(const xla_op arg) {
    BEGIN_PROTECT_OP
    return new XlaOp(RoundNearestEven(*arg));
    END_PROTECT_OP(arg)
}

xla_op op_log(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Log(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_log1p(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Log1p(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_logistic(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Logistic(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_sign(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Sign(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_clz(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Clz(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_cos(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Cos(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_sin(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Sin(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_tanh(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Tanh(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_real(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Real(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_imag(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Imag(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_conj(const xla_op arg) {
    BEGIN_PROTECT_OP
    return new XlaOp(Conj(*arg));
    END_PROTECT_OP(arg)
}

xla_op op_square(const xla_op arg) {
    BEGIN_PROTECT_OP
    return new XlaOp(Square(*arg));
    END_PROTECT_OP(arg)
}

xla_op op_sqrt(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Sqrt(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_rsqrt(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Rsqrt(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_cbrt(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Cbrt(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_is_finite(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(IsFinite(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_neg(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Neg(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_lower_triangle(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(LowerTriangle(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_upper_triangle(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(UpperTriangle(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_erf(const xla_op arg) {
    BEGIN_PROTECT_OP
    return new XlaOp(Erf(*arg));
    END_PROTECT_OP(arg)
}

xla_op op_einsum1(const xla_op arg, const char *config) {
  BEGIN_PROTECT_OP
  return new XlaOp(Einsum(*arg, config));
  END_PROTECT_OP(arg)
}

xla_op op_einsum2(const xla_op arg1, const xla_op arg2, const char *config) {
  BEGIN_PROTECT_OP
  return new XlaOp(Einsum(*arg1, *arg2, config));
  END_PROTECT_OP(arg1)
}

xla_op op_copy(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(Copy(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_clone(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(*arg);
  END_PROTECT_OP(arg)
}

xla_op op_zeros_like(const xla_op arg) {
  BEGIN_PROTECT_OP
  return new XlaOp(ZerosLike(*arg));
  END_PROTECT_OP(arg)
}

xla_op op_zero_like(const xla_op arg) {
  BEGIN_PROTECT_OP
  const Shape *shape = arg->builder()->GetShapePtr(*arg).value();
  return new XlaOp(Zero(arg->builder(), shape->element_type()));
  END_PROTECT_OP(arg)
}

xla_op op_reshape(const xla_op arg, size_t dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  return new XlaOp(Reshape(*arg, absl::Span<const int64_t>(ds, dsize)));
  END_PROTECT_OP(arg)
}

xla_op op_dynamic_reshape(const xla_op arg, size_t n_ops, const xla_op *ds,
                          size_t n_new_size_bounds, const int64_t *new_size_bounds,
                          const bool *dims_are_dynamic) {
    BEGIN_PROTECT_OP
        std::vector<XlaOp> vec_dim_sizes;
        for (size_t i = 0; i < n_ops; ++i) {
            vec_dim_sizes.push_back(*ds[i]);
        }

        std::vector<bool> vec_dims_are_dynamic;
        for (size_t i = 0; i < n_ops; ++i) {
            vec_dims_are_dynamic.push_back(dims_are_dynamic[i]);
        }

        return new XlaOp(
                DynamicReshape(*arg,
                               absl::Span<const XlaOp>(vec_dim_sizes),
                               absl::Span<const int64_t>(new_size_bounds, n_new_size_bounds),
                               vec_dims_are_dynamic));
    END_PROTECT_OP(arg)
}



xla_op op_broadcast(const xla_op arg, size_t dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  return new XlaOp(Broadcast(*arg, absl::Span<const int64_t>(ds, dsize)));
  END_PROTECT_OP(arg)
}

xla_op op_broadcast_in_dim(const xla_op arg, size_t out_dsize,
                           const int64_t *out_ds, size_t broadcast_dsize,
                           const int64_t *broadcast_ds) {
  BEGIN_PROTECT_OP
  return new XlaOp(
      BroadcastInDim(*arg, absl::Span<const int64_t>(out_ds, out_dsize),
                     absl::Span<const int64_t>(broadcast_ds, broadcast_dsize)));
  END_PROTECT_OP(arg)
}

xla_op op_collapse(const xla_op arg, size_t dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  return new XlaOp(Collapse(*arg, absl::Span<const int64_t>(ds, dsize)));
  END_PROTECT_OP(arg)
}

xla_op op_transpose(const xla_op arg, size_t dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  return new XlaOp(Transpose(*arg, absl::Span<const int64_t>(ds, dsize)));
  END_PROTECT_OP(arg)
}

xla_op op_clamp(const xla_op arg1, const xla_op arg2, const xla_op arg3) {
  BEGIN_PROTECT_OP
  return new XlaOp(Clamp(*arg1, *arg2, *arg3));
  END_PROTECT_OP(arg1)
}

xla_op op_select(const xla_op arg1, const xla_op arg2, const xla_op arg3) {
  BEGIN_PROTECT_OP
  return new XlaOp(Select(*arg1, *arg2, *arg3));
  END_PROTECT_OP(arg1)
}

xla_op op_call(const xla_builder b, const xla_computation f, size_t n_ops, const xla_op *args) {
    BEGIN_PROTECT_OP
        std::vector<XlaOp> args_;
        for (size_t i = 0; i < n_ops; ++i) {
            args_.push_back(*args[i]);
        }
        return new XlaOp(Call(b, *f, absl::Span<const XlaOp>(args_)));
    END_PROTECT_OP_B(b)
}

xla_op op_map(const xla_builder b, size_t n_ops, const xla_op *ops, const xla_computation f,
              size_t n_dims, const int64_t *dims, size_t n_static_ops, const xla_op *static_ops) {
    BEGIN_PROTECT_OP
        std::vector<XlaOp> ops_;
        for (size_t i = 0; i < n_ops; ++i) {
            ops_.push_back(*ops[i]);
        }
        std::vector<XlaOp> static_ops_;
        for (size_t i = 0; i < n_static_ops; ++i) {
            static_ops_.push_back(*static_ops[i]);
        }
        return new XlaOp(Map(b, absl::Span<const XlaOp>(ops_), *f,
                              absl::Span<const int64_t>(dims, n_dims),
                              absl::Span<const XlaOp>(static_ops_)));
    END_PROTECT_OP_B(b)
}

xla_op op_rng_uniform(const xla_op arg1, const xla_op arg2, int pr_type,
                      int dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  auto shape = ShapeUtil::MakeShape((PrimitiveType)pr_type,
                                    absl::Span<const int64_t>(ds, dsize));
  return new XlaOp(RngUniform(*arg1, *arg2, shape));
  END_PROTECT_OP(arg1)
}

xla_op op_rng_normal(const xla_op arg1, const xla_op arg2, int pr_type,
                     int dsize, const int64_t *ds) {
  BEGIN_PROTECT_OP
  auto shape = ShapeUtil::MakeShape((PrimitiveType)pr_type,
                                    absl::Span<const int64_t>(ds, dsize));
  return new XlaOp(RngNormal(*arg1, *arg2, shape));
  END_PROTECT_OP(arg1)
}

xla_op op_pad(const xla_op arg,
              const xla_op padding_value,
              size_t n_dims,
              const int64_t *edge_low,
              const int64_t *edge_high,
              const int64_t *interior) {
    BEGIN_PROTECT_OP
        PaddingConfig config;
        for (size_t i = 0; i < n_dims; ++i) {
            auto dim = config.add_dimensions();
            dim->set_edge_padding_low(edge_low[i]);
            dim->set_edge_padding_high(edge_high[i]);
            dim->set_interior_padding(interior[i]);
        }
        return new XlaOp(Pad(*arg, *padding_value, config));
    END_PROTECT_OP(arg)
}

xla_op op_pad_in_dim(const xla_op arg, const xla_op padding_value,
                     int64_t dinmo, int64_t pad_lo, int64_t pad_hi) {
    BEGIN_PROTECT_OP
    return new XlaOp(PadInDim(*arg, *padding_value, dinmo, pad_lo, pad_hi));
    END_PROTECT_OP(arg)
}


xla_op op_slice(const xla_op arg, size_t start_dsize, const int64_t *start_ds,
                size_t limit_dsize, const int64_t *limit_ds,
                size_t stride_dsize, const int64_t *stride_ds) {
    BEGIN_PROTECT_OP
    return new XlaOp(Slice(*arg, absl::Span<const int64_t>(start_ds, start_dsize),
                           absl::Span<const int64_t>(limit_ds, limit_dsize),
                           absl::Span<const int64_t>(stride_ds, stride_dsize)));
    END_PROTECT_OP(arg)
}

xla_op op_slice_in_dim(const xla_op arg, int64_t start, int64_t stop,
                       int64_t stride, int64_t dim) {
  BEGIN_PROTECT_OP
  return new XlaOp(SliceInDim(*arg, start, stop, stride, dim));
  END_PROTECT_OP(arg)
}

xla_op op_dynamic_slice(const xla_op arg, size_t n_ops,
                        const xla_op *start_indices,
                        size_t slice_dsize, const int64_t *slice_ds) {
    BEGIN_PROTECT_OP
        std::vector<XlaOp> indices;
        for (size_t i = 0; i < n_ops; ++i) {
            indices.push_back(*start_indices[i]);
        }
        return new XlaOp(
                DynamicSlice(*arg, absl::Span<const XlaOp>(indices),
                        absl::Span<const int64_t>(slice_ds, slice_dsize)));
    END_PROTECT_OP(arg)
}

xla_op op_dynamic_update_slice(const xla_op arg, const xla_op update,
                               size_t n_ops, const xla_op *start_indices) {
    BEGIN_PROTECT_OP
        std::vector<XlaOp> indices;
        for (size_t i = 0; i < n_ops; ++i) {
            indices.push_back(*start_indices[i]);
        }
        return new XlaOp(DynamicUpdateSlice(*arg, *update, absl::Span<const XlaOp>(indices)));
    END_PROTECT_OP(arg)
}

xla_op op_concat_in_dim(const xla_op arg, const xla_op *args, size_t nargs,
                        int64_t dim) {
  BEGIN_PROTECT_OP
  std::vector<XlaOp> args_ = {*arg};
  for (size_t i = 0; i < nargs; ++i) {
    args_.push_back(*args[i]);
  }
  return new XlaOp(
      ConcatInDim(arg->builder(), absl::Span<const XlaOp>(args_), dim));
  END_PROTECT_OP(arg)
}

xla_op op_tuple(const xla_builder b, const xla_op *args, size_t nargs) {
  BEGIN_PROTECT_OP
  std::vector<XlaOp> args_;
  for (size_t i = 0; i < nargs; ++i) {
    args_.push_back(*args[i]);
  }
  return new XlaOp(Tuple(b, absl::Span<const XlaOp>(args_)));
  END_PROTECT_OP_B(b)
}

xla_op op_get_tuple_element(const xla_op arg, int64_t index) {
  BEGIN_PROTECT_OP
  return new XlaOp(GetTupleElement(*arg, index));
  END_PROTECT_OP(arg)
}

xla_op op_gather(const xla_op arg1, const xla_op arg2,
                 const int64_t *offset_dims, size_t noffset_dims,
                 const int64_t *collapsed_slice_dims,
                 size_t ncollapsed_slice_dims, const int64_t *start_index_map,
                 size_t nstart_index_map, const int64_t *set_index_vector_dim,
                 const int64_t *slice_sizes, size_t nslice_sizes) {
  BEGIN_PROTECT_OP
  GatherDimensionNumbers dnums;
  for (size_t i = 0; i < noffset_dims; ++i) {
    dnums.add_offset_dims(offset_dims[i]);
  }
  for (size_t i = 0; i < ncollapsed_slice_dims; ++i) {
    dnums.add_collapsed_slice_dims(collapsed_slice_dims[i]);
  }
  for (size_t i = 0; i < nstart_index_map; ++i) {
    dnums.add_start_index_map(start_index_map[i]);
  }
  if (set_index_vector_dim) {
    dnums.set_index_vector_dim(*set_index_vector_dim);
  }
  auto ss = absl::Span<const int64_t>(slice_sizes, nslice_sizes);
  return new XlaOp(Gather(*arg1, *arg2, dnums, ss));
  END_PROTECT_OP(arg1)
}

xla_op op_scatter(size_t n_ops,
               const xla_op *operands,
               const xla_op scatter_indices,
               size_t n_updates,
               const xla_op *updates,
               const xla_computation comp,
               size_t n_update_window_dims,
               const int64_t *update_window_dims,
               size_t n_inserted_window_dims,
               const int64_t *inserted_window_dims,
               size_t n_scatter_dims_to_operand_dims,
               const int64_t *scatter_dims_to_operand_dims,
               int64_t index_vector_dim
               ) {
    BEGIN_PROTECT_OP
    std::vector<XlaOp> operands_;
    for (size_t i = 0; i < n_ops; ++i) {
        operands_.push_back(*operands[i]);
    }
    std::vector<XlaOp> updates_;
    for (size_t i = 0; i < n_updates; ++i) {
        updates_.push_back(*updates[i]);
    }
    ScatterDimensionNumbers dnums;
    for (size_t i = 0; i < n_update_window_dims; ++i) {
        dnums.add_update_window_dims(update_window_dims[i]);
    }
    for (size_t i = 0; i < n_inserted_window_dims; ++i) {
        dnums.add_inserted_window_dims(inserted_window_dims[i]);
    }
    for (size_t i = 0; i < n_scatter_dims_to_operand_dims; ++i) {
        dnums.add_scatter_dims_to_operand_dims(scatter_dims_to_operand_dims[i]);
    }
    dnums.set_index_vector_dim(index_vector_dim);
    return new XlaOp(Scatter(operands_, *scatter_indices, updates_, *comp, dnums));
    END_PROTECT_OP(scatter_indices)
}

xla_op op_convert_element_type(const xla_op arg, int pr_type) {
  BEGIN_PROTECT_OP
  return new XlaOp(ConvertElementType(*arg, (PrimitiveType)pr_type));
  END_PROTECT_OP(arg)
}

xla_op op_dimensions_size(const xla_op arg, int64_t dim) {
  BEGIN_PROTECT_OP
  return new XlaOp(GetDimensionSize(*arg, dim));
  END_PROTECT_OP(arg)
}

xla_op op_reduce(const xla_op arg, const xla_op init,
                 const xla_computation comp, const int64_t *dims,
                 size_t ndims) {
  BEGIN_PROTECT_OP
  return new XlaOp(
      Reduce(*arg, *init, *comp, absl::Span<const int64_t>(dims, ndims)));
  END_PROTECT_OP(arg)
}

xla_op op_internal_error(const xla_builder b, const char *error) {
  BEGIN_PROTECT_OP
  return new XlaOp(b->ReportError(tsl::errors::Internal(error)));
  END_PROTECT_OP_B(b)
}

xla_op op_unknown_error(const xla_builder b, const char *error) {
  BEGIN_PROTECT_OP
  return new XlaOp(b->ReportError(tsl::errors::Unknown(error)));
  END_PROTECT_OP_B(b)
}

xla_op op_invalid_argument_error(const xla_builder b, const char *error) {
  BEGIN_PROTECT_OP
  return new XlaOp(b->ReportError(tsl::errors::InvalidArgument(error)));
  END_PROTECT_OP_B(b)
}

xla_op op_zero(const xla_builder b, int pr_type) {
  BEGIN_PROTECT_OP
  return new XlaOp(Zero(b, (PrimitiveType)pr_type));
  END_PROTECT_OP_B(b)
}

xla_op op_one(const xla_builder b, int pr_type) {
  BEGIN_PROTECT_OP
  return new XlaOp(One(b, (PrimitiveType)pr_type));
  END_PROTECT_OP_B(b)
}

xla_op op_min_value(const xla_builder b, int pr_type) {
  BEGIN_PROTECT_OP
  return new XlaOp(MinValue(b, (PrimitiveType)pr_type));
  END_PROTECT_OP_B(b)
}

xla_op op_max_value(const xla_builder b, int pr_type) {
  BEGIN_PROTECT_OP
  return new XlaOp(MaxValue(b, (PrimitiveType)pr_type));
  END_PROTECT_OP_B(b)
}

xla_op op_iota1(const xla_builder b, int pr_type, size_t sz) {
  BEGIN_PROTECT_OP
  return new XlaOp(Iota(b, (PrimitiveType)pr_type, (int64_t)sz));
  END_PROTECT_OP_B(b)
}

xla_op op_iota(const xla_builder b, int pr_type, size_t dsize,
               const int64_t *ds, int64_t increasing_dim) {
  BEGIN_PROTECT_OP
  auto shape = ShapeUtil::MakeShape((PrimitiveType)pr_type,
                                    absl::Span<const int64_t>(ds, dsize));
  return new XlaOp(Iota(b, shape, increasing_dim));
  END_PROTECT_OP_B(b)
}

xla_op op_while(const xla_computation cond, const xla_computation body,
                const xla_op init) {
  BEGIN_PROTECT_OP
  return new XlaOp(While(*cond, *body, *init));
  END_PROTECT_OP(init)
}

xla_op op_conditional(const xla_op pred, const xla_op true_op,
                      const xla_computation true_comp, const xla_op false_op,
                      const xla_computation false_comp) {
  BEGIN_PROTECT_OP
  return new XlaOp(
      Conditional(*pred, *true_op, *true_comp, *false_op, *false_comp));
  END_PROTECT_OP(pred)
}

Padding ParsePadding(const char* padding_config) {
    if (std::string(padding_config) == "same") {
        return Padding::kSame;
    }
    if (std::string(padding_config) == "valid") {
        return Padding::kValid;
    }
    throw std::runtime_error("Invalid padding config: " + std::string(padding_config));
}

xla_op op_conv(const xla_op lhs,
               const xla_op rhs,
               size_t n_strides,
               const int64_t *window_strides,
               const char *padding_config,
               int64_t feature_group_count,
               int64_t batch_group_count) {
    BEGIN_PROTECT_OP
    Padding padding = ParsePadding(padding_config);
    return new XlaOp(
            Conv(*lhs, *rhs, absl::Span<const int64_t>(window_strides, n_strides), padding, feature_group_count, batch_group_count));
    END_PROTECT_OP(lhs)
}

xla_op op_conv_general_dilated(const xla_op lhs,
                               const xla_op rhs,
                               size_t n_strides,
                               const int64_t *window_strides,
                               size_t n_padding_pairs,
                               const int64_t *padding_values,
                               size_t n_lhs_dilations,
                               const int64_t *lhs_dilations,
                               size_t n_rhs_dilations,
                               const int64_t *rhs_dilations,
                               const int64_t *ibdim,
                               const int64_t *ifdim,
                               size_t n_isdims,
                               const int64_t *isdims,
                               const int64_t *obdim,
                               const int64_t *ofdim,
                               size_t n_osdims,
                               const int64_t *osdims,
                               const int64_t *kifdim,
                               const int64_t *kofdim,
                               size_t n_ksdims,
                               const int64_t *ksdims,
                               int64_t feature_group_count,
                               int64_t batch_group_count) {
    BEGIN_PROTECT_OP
    std::vector<std::pair<int64_t, int64_t>> padding_pairs;
    for (size_t i = 0; i < 2 * n_padding_pairs; i += 2) {
        padding_pairs.emplace_back(padding_values[i], padding_values[i + 1]);
    }
    ConvolutionDimensionNumbers dnums;
    dnums.set_input_batch_dimension(*ibdim);
    dnums.set_input_feature_dimension(*ifdim);
    for (size_t i = 0; i < n_isdims; ++i) {
        dnums.add_input_spatial_dimensions(isdims[i]);
    }
    dnums.set_output_batch_dimension(*obdim);
    dnums.set_output_feature_dimension(*ofdim);
    for (size_t i = 0; i < n_osdims; ++i) {
        dnums.add_output_spatial_dimensions(osdims[i]);
    }
    dnums.set_kernel_input_feature_dimension(*kifdim);
    dnums.set_kernel_output_feature_dimension(*kofdim);
    for (size_t i = 0; i < n_ksdims; ++i) {
        dnums.add_kernel_spatial_dimensions(ksdims[i]);
    }
    return new XlaOp(
            ConvGeneralDilated(*lhs,
                               *rhs,
                               absl::Span<const int64_t>(window_strides, n_strides),
                               absl::Span<const std::pair<int64_t, int64_t>>(padding_pairs),
                               absl::Span<const int64_t>(lhs_dilations, n_lhs_dilations),
                               absl::Span<const int64_t>(rhs_dilations, n_rhs_dilations),
                               dnums,
                               feature_group_count,
                               batch_group_count));
    END_PROTECT_OP(lhs)
}


xla_op op_batch_norm_inference(const xla_op operand,
                               const xla_op scale,
                               const xla_op offset,
                               const xla_op mean,
                               const xla_op variance,
                               float epsilon,
                               int64_t feature_index) {
   BEGIN_PROTECT_OP
       return new XlaOp(BatchNormInference(*operand, *scale, *offset, *mean, *variance, epsilon, feature_index));
   END_PROTECT_OP(operand)
}

xla_builder op_builder(const xla_op arg) { return arg->builder(); }

int xla_op_valid(const xla_op op) { return op->valid(); }

void xla_op_free(xla_op o) { delete o; }

size_t shape_tuple_shapes_size(const shape s) { return s->tuple_shapes_size(); }

shape shape_tuple_shapes(const shape s, int i) {
  return (shape)&s->tuple_shapes(i);
}

int shape_dimensions_size(const shape s) { return s->dimensions_size(); }

int shape_element_type(const shape s) { return s->element_type(); }

int64_t shape_dimensions(const shape s, int i) { return s->dimensions(i); }

void shape_free(shape s) { delete s; }

status get_shape(const xla_builder b, const xla_op o, shape *out_shape) {
  ASSIGN_OR_RETURN_STATUS(shape, b->GetShape(*o));
  *out_shape = new Shape(shape);
  return nullptr;
}

status get_element_type(const xla_builder b, const xla_op o,
                        int *out_element_type) {
  ASSIGN_OR_RETURN_STATUS(shape, b->GetShapePtr(*o));
  *out_element_type = shape->element_type();
  return nullptr;
}

status get_dimensions_size(const xla_builder b, const xla_op o, int *out_rank) {
  ASSIGN_OR_RETURN_STATUS(shape, b->GetShapePtr(*o));
  *out_rank = shape->dimensions_size();
  return nullptr;
}

status get_dimensions(const xla_builder b, const xla_op o, size_t *out_dims) {
  ASSIGN_OR_RETURN_STATUS(shape, b->GetShapePtr(*o));
  size_t dim_size = shape->dimensions_size();
  for (size_t i = 0; i < dim_size; ++i) {
    out_dims[i] = shape->dimensions(i);
  }
  return nullptr;
}

status build(const xla_builder b, const xla_op o, xla_computation *output) {
  ASSIGN_OR_RETURN_STATUS(computation, b->Build(o));
  *output = new XlaComputation();
  **output = std::move(computation);
  return nullptr;
}

status compile(const pjrt_client client, const xla_computation computation,
               pjrt_loaded_executable *output) {
  CompileOptions options;
  ASSIGN_OR_RETURN_STATUS(executable,
                          (*client)->Compile(*computation, options));
  *output = executable.release();
  return nullptr;
}

status first_error(const xla_builder b) {
  MAYBE_RETURN_STATUS(b->first_error());
  return nullptr;
}

status get_current_status(const xla_builder b) {
  MAYBE_RETURN_STATUS(b->GetCurrentStatus());
  return nullptr;
}

status execute(const pjrt_loaded_executable exe, const literal *inputs,
               int ninputs, pjrt_buffer ***outputs) {
  auto client = exe->client();
  ExecuteOptions options;
  options.strict_shape_checking = false;
  std::vector<PjRtBuffer *> input_buffer_ptrs;
  PjRtDevice *device = client->devices()[0];
  for (int i = 0; i < ninputs; ++i) {
    ASSIGN_OR_RETURN_STATUS(buffer,
                            client->BufferFromHostLiteral(*inputs[i], device));
    // Wait for the transfer to have completed to avoid the literal potentially
    // getting out of scope before it has been transferred.
    MAYBE_RETURN_STATUS(buffer->GetReadyFuture().Await());
    input_buffer_ptrs.push_back(buffer.release());
  }
  ASSIGN_OR_RETURN_STATUS(results, exe->Execute({input_buffer_ptrs}, options));
  pjrt_buffer **out =
      (pjrt_buffer **)malloc((results.size() + 1) * sizeof(pjrt_buffer *));
  for (size_t i = 0; i < results.size(); ++i) {
    auto &replica_results = results[i];
    pjrt_buffer *per_replica_outputs = (pjrt_buffer *)malloc(
        (replica_results.size() + 1) * sizeof(pjrt_buffer));
    for (size_t j = 0; j < replica_results.size(); ++j) {
      per_replica_outputs[j] = replica_results[j].release();
    }
    per_replica_outputs[replica_results.size()] = nullptr;
    out[i] = per_replica_outputs;
  }
  out[results.size()] = nullptr;
  *outputs = out;
  return nullptr;
}

status execute_b(const pjrt_loaded_executable exe, const pjrt_buffer *inputs,
                 int ninputs, pjrt_buffer ***outputs) {
  auto client = exe->client();
  ExecuteOptions options;
  options.strict_shape_checking = false;
  std::vector<PjRtBuffer *> input_buffer_ptrs(inputs, inputs + ninputs);
  ASSIGN_OR_RETURN_STATUS(results, exe->Execute({input_buffer_ptrs}, options));
  pjrt_buffer **out =
      (pjrt_buffer **)malloc((results.size() + 1) * sizeof(pjrt_buffer *));
  for (size_t i = 0; i < results.size(); ++i) {
    auto &replica_results = results[i];
    pjrt_buffer *per_replica_outputs = (pjrt_buffer *)malloc(
        (replica_results.size() + 1) * sizeof(pjrt_buffer));
    for (size_t j = 0; j < replica_results.size(); ++j) {
      per_replica_outputs[j] = replica_results[j].release();
    }
    per_replica_outputs[replica_results.size()] = nullptr;
    out[i] = per_replica_outputs;
  }
  out[results.size()] = nullptr;
  *outputs = out;
  return nullptr;
}

literal literal_create_from_shape(int pr_type, const int64_t *dims,
                                  size_t ndims) {
  auto shape = ShapeUtil::MakeShape((PrimitiveType)pr_type,
                                    absl::Span<const int64_t>(dims, ndims));
  Literal l = Literal::CreateFromShape(shape);
  return new Literal(std::move(l));
}

literal literal_create_from_shape_and_data(int pr_type, const int64_t *dims,
                                           size_t ndims, const void *data,
                                           size_t data_len) {
  auto shape = ShapeUtil::MakeShape((PrimitiveType)pr_type,
                                    absl::Span<const int64_t>(dims, ndims));
  Literal l = Literal::CreateFromShape(shape);
  if (l.size_bytes() != data_len) {
    return nullptr;
  }
  memcpy(l.untyped_data(), data, data_len);
  return new Literal(std::move(l));
}

literal literal_clone(const literal l) {
  return new Literal(std::move(l->Clone()));
}

status literal_reshape(const literal l, const int64_t *dims, size_t ndims,
                       literal *output) {
  ASSIGN_OR_RETURN_STATUS(literal,
                          l->Reshape(absl::Span<const int64_t>(dims, ndims)));
  *output = new Literal(std::move(literal));
  return nullptr;
}

status literal_convert(const literal l, int pr_type, literal *output) {
  ASSIGN_OR_RETURN_STATUS(literal, l->Convert((PrimitiveType)pr_type));
  *output = new Literal(std::move(literal));
  return nullptr;
}

int64_t literal_element_count(const literal l) { return l->element_count(); }

int64_t literal_size_bytes(const literal l) { return l->size_bytes(); }

void literal_shape(const literal l, shape *out_shape) {
  *out_shape = new Shape(l->shape());
}

void literal_decompose_tuple(literal l, literal *outputs, size_t noutputs) {
  auto tuple = l->DecomposeTuple();
  for (int i = 0; i < std::min(noutputs, tuple.size()); ++i) {
    outputs[i] = new Literal(std::move(tuple[i]));
  }
}

int literal_element_type(const literal l) { return l->shape().element_type(); }

void literal_copy_to(const literal l, void *dst, size_t size_in_bytes) {
  std::memcpy(dst, l->untyped_data(), size_in_bytes);
}

void literal_copy_from(literal l, const void *src, size_t size_in_bytes) {
  std::memcpy(l->untyped_data(), src, size_in_bytes);
}

literal literal_make_tuple(const literal *l, size_t n) {
  Literal out = LiteralUtil::MakeTuple(absl::MakeSpan(l, n));
  return new Literal(std::move(out));
}

literal literal_make_tuple_owned(const literal *l, size_t n) {
  std::vector<xla::Literal> elems;
  for (size_t i = 0; i < n; ++i) {
    elems.push_back(std::move(*(l[i])));
  }
  Literal out = LiteralUtil::MakeTupleOwned(std::move(elems));
  return new Literal(std::move(out));
}

void literal_free(literal l) { delete l; }

void status_free(status s) { delete s; }

char *xla_computation_name(xla_computation c) {
  return strdup(std::string(c->name()).c_str());
}

void xla_computation_free(xla_computation c) { delete c; }

char *status_error_message(status s) {
  return strdup(s->error_message().c_str());
}

status hlo_module_proto_parse_and_return_unverified_module(
    const char *data, size_t len, hlo_module_proto *output) {
  ASSIGN_OR_RETURN_STATUS(
      hmp, ParseAndReturnUnverifiedModule(std::string(data, len)));
  *output = new HloModuleProto(hmp->ToProto());
  return nullptr;
}

status hlo_module_proto_parse_proto(const char *d, size_t len, bool binary,
                                    hlo_module_proto *output) {
  std::string data(d, len);
  HloSnapshot proto;
  if (binary) {
    if (!proto.ParseFromString(data) &&
        !proto.mutable_hlo()->ParseFromString(data) &&
        !proto.mutable_hlo()->mutable_hlo_module()->ParseFromString(data)) {
      return new Status(
          InvalidArgument("Failed to parse input as HLO protobuf binary"));
    }
  } else {
    if (!tsl::protobuf::TextFormat::ParseFromString(data, &proto) &&
        !tsl::protobuf::TextFormat::ParseFromString(data,
                                                    proto.mutable_hlo()) &&
        !tsl::protobuf::TextFormat::ParseFromString(
            data, proto.mutable_hlo()->mutable_hlo_module())) {
      return new Status(
          InvalidArgument("Failed to parse input as HLO protobuf text"));
    }
  }
  ASSIGN_OR_RETURN_STATUS(config, HloModule::CreateModuleConfigFromProto(
                                      proto.hlo().hlo_module(), {}));
  ASSIGN_OR_RETURN_STATUS(
      hmp, HloModule::CreateFromProto(proto.hlo().hlo_module(), config));
  *output = new HloModuleProto(hmp->ToProto());
  return nullptr;
}

status hlo_module_from_proto(const hlo_module_proto input_proto, hlo_module *output) {
    ASSIGN_OR_RETURN_STATUS(
            config, HloModule::CreateModuleConfigFromProto(*input_proto, {}));
    ASSIGN_OR_RETURN_STATUS(
            hm, HloModule::CreateFromProto(*input_proto, config));
    *output = hm.release();
    return nullptr;
}

hlo_computation hlo_module_entry_computation(const hlo_module module) {
    return module->entry_computation();
}

int64_t hlo_module_computation_count(const hlo_module module) {
    return module->computation_count();
}

int64_t hlo_module_instruction_count(const hlo_module module) {
    return module->instruction_count();
}

char* hlo_module_to_string(const hlo_module module) {
    std::string result = module->ToString();
    char* output = new char[result.length() + 1];
    std::strcpy(output, result.c_str());
    return output;
}

xla_computation xla_computation_from_hlo_module_proto(const hlo_module_proto p) {
  return new XlaComputation(*p);
}

void hlo_module_proto_free(hlo_module_proto p) { delete p; }

hlo_module_proto xla_computation_proto(const xla_computation c) {
  return new HloModuleProto(c->proto());
}
