#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#pragma GCC diagnostic ignored "-Wreturn-type"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#pragma GCC diagnostic pop
using namespace xla;

extern "C" {
typedef std::shared_ptr<PjRtClient> *pjrt_client;
typedef PjRtLoadedExecutable *pjrt_loaded_executable;
typedef PjRtDevice *pjrt_device;
typedef PjRtBuffer *pjrt_buffer;
typedef XlaBuilder *xla_builder;
typedef XlaOp *xla_op;
typedef Status *status;
typedef Shape *shape;
typedef Literal *literal;
typedef XlaComputation *xla_computation;
typedef HloModule *hlo_module;
typedef HloModuleProto *hlo_module_proto;
typedef HloComputation *hlo_computation;
#else
typedef struct _pjrt_client *pjrt_client;
typedef struct _pjrt_loaded_executable *pjrt_loaded_executable;
typedef struct _pjrt_device *pjrt_device;
typedef struct _pjrt_buffer *pjrt_buffer;
typedef struct _xla_builder *xla_builder;
typedef struct _xla_op *xla_op;
typedef struct _status *status;
typedef struct _shape *shape;
typedef struct _literal *literal;
typedef struct _xla_computation *xla_computation;
typedef struct _hlo_module *hlo_module;
typedef struct _hlo_module_proto *hlo_module_proto;
typedef struct _hlo_computation *hlo_computation;
#endif

status pjrt_cpu_client_create(pjrt_client *);
status pjrt_gpu_client_create(pjrt_client *, double, bool);
status pjrt_tpu_client_create(pjrt_client *, int);
void pjrt_client_free(pjrt_client);
int pjrt_client_device_count(pjrt_client);
int pjrt_client_addressable_device_count(pjrt_client);
void pjrt_client_devices(pjrt_client, pjrt_device *);
void pjrt_client_addressable_devices(pjrt_client, pjrt_device *);
char *pjrt_client_platform_name(pjrt_client);
char *pjrt_client_platform_version(pjrt_client);

void pjrt_loaded_executable_free(pjrt_loaded_executable);

int pjrt_device_id(pjrt_device);
int pjrt_device_process_index(pjrt_device);
int pjrt_device_local_hardware_id(pjrt_device);
status pjrt_device_transfer_to_infeed(pjrt_device, const literal);
status pjrt_device_transfer_from_outfeed(pjrt_device, literal);
char *pjrt_device_kind(pjrt_device);
char *pjrt_device_debug_string(pjrt_device);
char *pjrt_device_to_string(pjrt_device);

status pjrt_buffer_from_host_literal(const pjrt_client, const pjrt_device,
                                     const literal, pjrt_buffer *);
status pjrt_buffer_from_host_buffer(const pjrt_client, const pjrt_device,
                                    const void *, int, int, const int64_t *,
                                    pjrt_buffer *);
status pjrt_buffer_to_literal_sync(pjrt_buffer, literal *);
status pjrt_buffer_copy_raw_to_host_sync(pjrt_buffer, void *, size_t, size_t);
shape pjrt_buffer_on_device_shape(pjrt_buffer);
status pjrt_buffer_copy_to_device(pjrt_buffer, pjrt_device, pjrt_buffer *);
void pjrt_buffer_free(pjrt_buffer);

xla_builder xla_builder_create(const char *);
void xla_builder_free(xla_builder);

xla_op constant_literal(const xla_builder, const literal);
xla_op parameter(const xla_builder, int64_t, int, int, const int64_t *,
                 const char *);
xla_op parameter_s(const xla_builder, int64_t, const shape, const char *);
xla_op infeed(const xla_builder, int, int, const int64_t *, const char *);
void outfeed(const xla_op, int, int, const int64_t *, const char *);

// Ops
xla_op op_add(const xla_op, const xla_op);
xla_op op_sub(const xla_op, const xla_op);
xla_op op_mul(const xla_op, const xla_op);
xla_op op_div(const xla_op, const xla_op);
xla_op op_rem(const xla_op, const xla_op);
xla_op op_max(const xla_op, const xla_op);
xla_op op_min(const xla_op, const xla_op);
xla_op op_and(const xla_op, const xla_op);
xla_op op_or(const xla_op, const xla_op);
xla_op op_xor(const xla_op, const xla_op);
xla_op op_atan2(const xla_op, const xla_op);
xla_op op_pow(const xla_op, const xla_op);
xla_op op_dot(const xla_op, const xla_op);
xla_op op_dot_general(const xla_op, const xla_op, const int64_t *, size_t,
                      const int64_t *, size_t, const int64_t *, size_t,
                      const int64_t *, size_t);
xla_op op_eq(const xla_op, const xla_op);
xla_op op_ne(const xla_op, const xla_op);
xla_op op_ge(const xla_op, const xla_op);
xla_op op_gt(const xla_op, const xla_op);
xla_op op_le(const xla_op, const xla_op);
xla_op op_lt(const xla_op, const xla_op);
xla_op op_shift_left(const xla_op, const xla_op);
xla_op op_shift_right_arith(const xla_op, const xla_op);
xla_op op_shift_right_logic(const xla_op, const xla_op);
xla_op op_population_count(const xla_op);
xla_op op_not(const xla_op);
xla_op op_abs(const xla_op);
xla_op op_exp(const xla_op);
xla_op op_expm1(const xla_op);
xla_op op_floor(const xla_op);
xla_op op_ceil(const xla_op);
xla_op op_round(const xla_op);
xla_op op_round_nearest_even(const xla_op);
xla_op op_log(const xla_op);
xla_op op_log1p(const xla_op);
xla_op op_logistic(const xla_op);
xla_op op_sign(const xla_op);
xla_op op_clz(const xla_op);
xla_op op_cos(const xla_op);
xla_op op_sin(const xla_op);
xla_op op_tanh(const xla_op);
xla_op op_real(const xla_op);
xla_op op_imag(const xla_op);
xla_op op_conj(const xla_op);
xla_op op_square(const xla_op);
xla_op op_sqrt(const xla_op);
xla_op op_rsqrt(const xla_op);
xla_op op_cbrt(const xla_op);
xla_op op_is_finite(const xla_op);
xla_op op_neg(const xla_op);
xla_op op_lower_triangle(const xla_op);
xla_op op_upper_triangle(const xla_op);
xla_op op_erf(const xla_op);
xla_op op_einsum1(const xla_op, const char *);
xla_op op_einsum2(const xla_op, const xla_op, const char *);
xla_op op_copy(const xla_op);
xla_op op_clone(const xla_op);
xla_op op_zeros_like(const xla_op);
xla_op op_zero_like(const xla_op);
xla_op op_zero(const xla_builder, int);
xla_op op_one(const xla_builder, int);
xla_op op_min_value(const xla_builder, int);
xla_op op_max_value(const xla_builder, int);
xla_op op_reshape(const xla_op, size_t, const int64_t *);
xla_op op_dynamic_reshape(const xla_op, size_t, const xla_op *, size_t, const int64_t *, const bool *);
xla_op op_broadcast(const xla_op, size_t, const int64_t *);
xla_op op_broadcast_in_dim(const xla_op, size_t, const int64_t *, size_t,
                           const int64_t *);
xla_op op_collapse(const xla_op, size_t, const int64_t *);
xla_op op_transpose(const xla_op, size_t, const int64_t *);
xla_op op_clamp(const xla_op, const xla_op, const xla_op);
xla_op op_select(const xla_op, const xla_op, const xla_op);
xla_op op_call(const xla_builder, const xla_computation, size_t, const xla_op *);
xla_op op_map(const xla_builder, size_t, const xla_op *, const xla_computation, size_t, const int64_t *, size_t, const xla_op *);
xla_op op_rng_uniform(const xla_op, const xla_op, int, int, const int64_t *);
xla_op op_rng_normal(const xla_op, const xla_op, int, int, const int64_t *);
xla_op op_pad(const xla_op, const xla_op, size_t, const int64_t *, const int64_t *, const int64_t *);
xla_op op_pad_in_dim(const xla_op, const xla_op, int64_t, int64_t, int64_t);
xla_op op_slice(const xla_op, size_t, const int64_t *, size_t, const int64_t *, size_t, const int64_t *);
xla_op op_slice_in_dim(const xla_op, int64_t, int64_t, int64_t, int64_t);
xla_op op_dynamic_slice(const xla_op, size_t, const xla_op *, size_t, const int64_t *);
xla_op op_dynamic_update_slice(const xla_op, const xla_op, size_t, const xla_op *);
xla_op op_concat_in_dim(const xla_op, const xla_op *, size_t, int64_t);
xla_op op_tuple(const xla_builder, const xla_op *, size_t);
xla_op op_get_tuple_element(const xla_op, int64_t);
xla_op op_gather(const xla_op, const xla_op, const int64_t *, size_t,
                 const int64_t *, size_t, const int64_t *, size_t,
                 const int64_t *, const int64_t *, size_t);
xla_op op_scatter(size_t, const xla_op *, const xla_op, size_t, const xla_op *, const xla_computation,
                  size_t, const int64_t *, size_t, const int64_t *, size_t, const int64_t *, int64_t);
xla_op op_convert_element_type(const xla_op, int);
xla_op op_dimensions_size(const xla_op, int64_t);
xla_op op_reduce(const xla_op, const xla_op, const xla_computation,
                 const int64_t *, size_t);
xla_op op_internal_error(const xla_builder, const char *);
xla_op op_unknown_error(const xla_builder, const char *);
xla_op op_invalid_argument_error(const xla_builder, const char *);
xla_op op_iota1(const xla_builder, int, size_t);
xla_op op_iota(const xla_builder, int, size_t, const int64_t *, int64_t);
xla_op op_while(const xla_computation, const xla_computation, const xla_op);
xla_op op_conditional(const xla_op, const xla_op, const xla_computation,
                      const xla_op, const xla_computation);
xla_op op_conv(const xla_op, const xla_op, size_t, const int64_t *, const char*, int64_t, int64_t);
xla_op op_conv_general_dilated(const xla_op, const xla_op,
                               size_t, const int64_t *,
                               size_t, const int64_t *,
                               size_t, const int64_t *,
                               size_t, const int64_t *,
                               const int64_t *,
                               const int64_t *,
                               size_t, const int64_t *,
                               const int64_t *,
                               const int64_t *,
                               size_t, const int64_t *,
                               const int64_t *,
                               const int64_t *,
                               size_t, const int64_t *,
                               int64_t, int64_t);
xla_op op_batch_norm_inference(const xla_op,
                               const xla_op,
                               const xla_op,
                               const xla_op,
                               const xla_op,
                               float,
                               int64_t);

xla_builder op_builder(const xla_op);

int xla_op_valid(const xla_op);
void xla_op_free(xla_op);

int shape_dimensions_size(const shape);
size_t shape_tuple_shapes_size(const shape);
shape shape_tuple_shapes(const shape, int);
int shape_element_type(const shape);
int64_t shape_dimensions(const shape, int);
void shape_free(shape);
shape make_shape_array(int, size_t, const int64_t *);
shape make_shape_tuple(size_t, const shape *);

status get_shape(const xla_builder, const xla_op, shape *);
status get_element_type(const xla_builder, const xla_op, int *);
status get_dimensions_size(const xla_builder, const xla_op, int *);
status get_dimensions(const xla_builder, const xla_op, size_t *);

status build(const xla_builder, const xla_op, xla_computation *);
status compile(const pjrt_client, const xla_computation,
               pjrt_loaded_executable *);
status execute(const pjrt_loaded_executable, const literal *, int,
               pjrt_buffer ***);
status execute_b(const pjrt_loaded_executable, const pjrt_buffer *, int,
                 pjrt_buffer ***);
status first_error(const xla_builder);
status get_current_status(const xla_builder);

literal literal_create_from_shape(int, const int64_t *, size_t);
literal literal_create_from_shape_and_data(int, const int64_t *, size_t,
                                           const void *, size_t);
literal literal_clone(const literal);
status literal_reshape(const literal, const int64_t *, size_t, literal *);
status literal_convert(const literal, int, literal *);
int64_t literal_element_count(const literal);
int literal_element_type(const literal);
void literal_shape(const literal, shape *);
void literal_decompose_tuple(literal, literal *, size_t);
int64_t literal_size_bytes(const literal);
void literal_copy_to(const literal, void *, size_t);
void literal_copy_from(literal, const void *, size_t);
literal literal_make_tuple(const literal *, size_t);
literal literal_make_tuple_owned(const literal *, size_t);
void literal_free(literal);

status hlo_module_proto_parse_and_return_unverified_module(const char *, size_t,
                                                           hlo_module_proto *);
status hlo_module_proto_parse_proto(const char *, size_t, bool,
                                    hlo_module_proto *);
status hlo_module_from_proto(const hlo_module_proto,  hlo_module *);

hlo_computation hlo_module_entry_computation(const hlo_module);
int64_t hlo_module_computation_count(const hlo_module);
int64_t hlo_module_instruction_count(const hlo_module);
char *hlo_module_to_string(const hlo_module);

xla_computation xla_computation_from_hlo_module_proto(const hlo_module_proto);
void hlo_module_proto_free(hlo_module_proto);

char *xla_computation_name(xla_computation);
hlo_module_proto xla_computation_proto(const xla_computation);
void xla_computation_free(xla_computation);

void status_free(status);
char *status_error_message(status);

#define FOR_EACH_NATIVE_TYPE(_)                                                \
  _(bool, PRED)                                                                \
  _(int8_t, S8)                                                                \
  _(int16_t, S16)                                                              \
  _(int32_t, S32)                                                              \
  _(int64_t, S64)                                                              \
  _(uint8_t, U8)                                                               \
  _(uint16_t, U16)                                                             \
  _(uint32_t, U32)                                                             \
  _(uint64_t, U64)                                                             \
  _(float, F32)                                                                \
  _(double, F64)

#define CONST_OP_R01(native_type, primitive_type)                              \
  xla_op constant_r0_##native_type(const xla_builder, native_type);            \
  xla_op constant_r1c_##native_type(const xla_builder, native_type, size_t);   \
  xla_op constant_r1_##native_type(const xla_builder, const native_type *,     \
                                   size_t);                                    \
  literal create_r0_##native_type(native_type);                                \
  literal create_r1_##native_type(const native_type *, size_t);                \
  native_type literal_get_first_element_##native_type(const literal);

FOR_EACH_NATIVE_TYPE(CONST_OP_R01)
#undef CONST_OP_R01

#ifdef __cplusplus
}
#endif
