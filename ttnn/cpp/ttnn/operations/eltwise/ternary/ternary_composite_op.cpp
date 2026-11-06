// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <reflect>

#include "ternary_composite_op.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"

namespace ttnn {

namespace operations::ternary {

// addcmul(input,tensor1,tensor2,value)=input+value×tensor1×tensor2
Tensor _addcmul(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(
        input_a.storage_type() == StorageType::DEVICE && input_b.storage_type() == StorageType::DEVICE &&
            input_c.storage_type() == StorageType::DEVICE,
        "Ternary operation requires input tensors to be on Device.");

    // Associate as (value * input_b) * input_c, matching both the LLK kernel
    // (ckernel_sfpu_addcmul.h) and the registered golden torch.addcmul, which computes
    // self + scalar_val * t1_val * t2_val left-to-right. Scaling by `value` first (rather
    // than last) avoids spurious overflow/underflow when `value` is chosen to keep the
    // product in range (e.g. Adam's addcmul(avg_sq, grad, grad, value=1-beta2)).
    Tensor t_scaled = ttnn::multiply(input_b, value, std::nullopt, output_mem_config);
    Tensor t_factor = ttnn::multiply(t_scaled, input_c, std::nullopt, output_mem_config);
    t_scaled.deallocate();
    Tensor result = ttnn::add(input_a, t_factor, std::nullopt, output_mem_config);
    if (output_tensor.has_value()) {
        ttnn::assign(result, output_tensor.value());
        return output_tensor.value();
    }
    return result;
}

// addcdiv(input,tensor1,tensor2,value)=input+(value×tensor1)/tensor2
Tensor _addcdiv(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {

    TT_FATAL(
        input_a.storage_type() == StorageType::DEVICE &&
            input_b.storage_type() == StorageType::DEVICE &&
            input_c.storage_type() == StorageType::DEVICE,
        "Ternary operation requires input tensors to be on Device.");

    auto write_output = [&](const Tensor& result) {
        if (!output_tensor.has_value()) {
            return result;
        }
        ttnn::assign(result, output_tensor.value());
        return output_tensor.value();
    };

    Tensor t_factor =
        ttnn::multiply(input_b, value, std::nullopt, output_mem_config);

    Tensor t_div = ttnn::div(
        t_factor, input_c, false, std::nullopt, std::nullopt, output_mem_config);

    // Preserve the existing FP32 implementation.
    if (input_a.dtype() == DataType::FLOAT32) {
        Tensor result = ttnn::add(input_a, t_div, std::nullopt, output_mem_config);
        t_factor.deallocate();
        t_div.deallocate();
        return write_output(result);
    }

    // Detect zero divisor, including negative zero.
    Tensor zero_c = ttnn::eqz(input_c, output_mem_config);

    // Determine the sign of the numerator.
    Tensor sign_b = ttnn::sign(input_b, output_mem_config);

    // Preserve the sign of the zero divisor.
    Tensor sign_c = ttnn::where(
        ttnn::signbit(input_c, output_mem_config),
        -1.0f,
        1.0f,
        output_mem_config);

    Tensor sign_or_one = ttnn::where(
        ttnn::eqz(input_b, output_mem_config),
        1.0f,
        sign_b,
        output_mem_config);

    // Construct the signed pole infinity.
    const float signed_inf =
        value * std::numeric_limits<float>::infinity();

    Tensor pole_sign = ttnn::multiply(
        sign_or_one, sign_c, std::nullopt, output_mem_config);

    Tensor t_inf = ttnn::multiply(
        pole_sign, signed_inf, std::nullopt, output_mem_config);

    // 0/0 and NaN/0 must produce NaN.
    const float nan_value = std::numeric_limits<float>::quiet_NaN();

    Tensor t_pole = ttnn::where(
        ttnn::eqz(input_b, output_mem_config),
        nan_value,
        t_inf,
        output_mem_config);

    t_pole = ttnn::where(
        ttnn::isnan(input_b, output_mem_config),
        nan_value,
        t_pole,
        output_mem_config);

    // Fix the quotient BEFORE adding input_a.
    Tensor t_fixed_div = ttnn::where(
        zero_c,
        t_pole,
        t_div,
        output_mem_config);

    // Release intermediate tensors.
    t_factor.deallocate();
    t_div.deallocate();
    sign_b.deallocate();
    sign_c.deallocate();
    sign_or_one.deallocate();
    pole_sign.deallocate();
    t_inf.deallocate();
    t_pole.deallocate();
    zero_c.deallocate();

    // NaN propagation and opposite infinities are now handled by add.
    Tensor result = ttnn::add(
        input_a, t_fixed_div, std::nullopt, output_mem_config);

    t_fixed_div.deallocate();

    return write_output(result);
}


// Fallback composite implementation for lerp (with scalar weight)
// lerp(input, end, weight) = start + weight * (end - start)
Tensor _lerp_overload(
    const Tensor& input,
    const Tensor& end,
    float weight,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(input.dtype() == end.dtype(), "Expected the same dtype as start (input), for end");

    // Validate output tensor dtype if provided
    if (output_tensor.has_value()) {
        auto input_dtype = input.dtype();
        auto output_dtype = output_tensor->dtype();

        bool valid_dtype = (output_dtype == input_dtype) ||
                           (output_dtype == DataType::FLOAT32 && input_dtype == DataType::BFLOAT16) ||
                           (output_dtype == DataType::BFLOAT16 && input_dtype == DataType::FLOAT32);

        TT_FATAL(
            valid_dtype,
            "Output tensor dtype must match input dtype, or be FLOAT32 when inputs are BFLOAT16, or be BFLOAT16 when "
            "inputs are FLOAT32. "
            "Got input dtype: {}, output dtype: {}",
            input_dtype,
            output_dtype);
    }

    Tensor t_diff = ttnn::subtract(end, input, std::nullopt, output_mem_config);
    Tensor t_mul = ttnn::multiply(t_diff, weight, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input, t_mul, std::nullopt, output_mem_config);

    // Handle optional output tensor
    if (output_tensor.has_value()) {
        ttnn::assign(result, output_tensor.value());
        return output_tensor.value();
    }

    return result;
}

// LLK implementation for lerp (with tensor weight)
Tensor _lerp(
    const Tensor& input,
    const Tensor& end,
    const Tensor& weight,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(input.dtype() == end.dtype(), "Expected the same dtype as start (input), for end");
    TT_FATAL(input.dtype() == weight.dtype(), "Expected the same dtype as start (input), for weight");

    // Validate output tensor dtype if provided
    if (output_tensor.has_value()) {
        auto input_dtype = input.dtype();
        auto output_dtype = output_tensor->dtype();

        bool valid_dtype = (output_dtype == input_dtype) ||
                           (output_dtype == DataType::FLOAT32 && input_dtype == DataType::BFLOAT16) ||
                           (output_dtype == DataType::BFLOAT16 && input_dtype == DataType::FLOAT32);

        TT_FATAL(
            valid_dtype,
            "Output tensor dtype must match input dtype, or be FLOAT32 when inputs are BFLOAT16, or be BFLOAT16 when "
            "inputs are FLOAT32. "
            "Got input dtype: {}, output dtype: {}",
            input_dtype,
            output_dtype);
    }

    Tensor t_diff = ttnn::multiply(
        ttnn::subtract(end, input, std::nullopt, output_mem_config), weight, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input, t_diff, std::nullopt, output_mem_config);

    // Handle optional output tensor
    if (output_tensor.has_value()) {
        ttnn::assign(result, output_tensor.value());
        return output_tensor.value();
    }

    return result;
}

}  // namespace operations::ternary

// Fallback composite implementations for mac: y = a * b + c.
// Used when the native LLK path cannot serve the request (unsupported broadcast,
// block-float subtile broadcast, or integer dtypes, which the SFPU mac path does
// not handle - it computes in the FP32 SFPU accumulator).
Tensor _mac(const Tensor& a, const Tensor& b, const Tensor& c, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
}

// TTS: a * b + scalar
Tensor _mac(const Tensor& a, const Tensor& b, float c, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
}

// TST: a * scalar + c
Tensor _mac(const Tensor& a, float b, const Tensor& c, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
}

// TSS: a * scalar1 + scalar2
Tensor _mac(const Tensor& a, float b, float c, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
}

}  // namespace ttnn
