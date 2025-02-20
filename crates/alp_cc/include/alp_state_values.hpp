#include "alp.hpp"
#include <memory>
#include <iostream>
#include <rust/cxx.h>
#include <limits>

// This is wrapper method for getting attributes from ALP's complex struct state
// The main use from this library is that we pass the ALP state
// during the different stages of compression and receive
// an attribute that we need
// Atributes that can be obtained from this 

#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wimplicit-int-conversion"


namespace alp_utils {

    struct float_state : public alp::state<float> {};

    // new_state
    std::unique_ptr<float_state> new_float_state() {
        // returns new ALP state
        // state stt = state(); 
        std::unique_ptr<float_state> stt = std::make_unique<float_state>();
        // sending back the memory address of stt
        // rust side will in turn pass the mutable pointer to c++ side
        // c++ side upon receiving pointer, will read state object and work on it. 
        return stt;
    }


    // wrapper function that receives rust slices and calls ALP alp_sample
    static inline void alp_init(
        rust::Slice<const float> data_column, 
        size_t column_offset, 
        size_t tuples_count, 
        float_state& stt) {
            // create dynamic buffer for sample_arr
            std::unique_ptr<float> sample_arr (new float[alp::config::VECTOR_SIZE]);
            // call ALP's alp_sample
            alp::encoder<float>::init(
                data_column.data(), 
                column_offset,
                tuples_count,
                sample_arr.get(),
                stt
            );
            // std.cout << 
            // stt.scheme = alp::Scheme::ALP;
            // std::cout <<"alp_init: The schema:" << get_state_scheme(stt) << std::endl;
        }


    static uint16_t alp_encode(
        rust::Slice<const float> input_vector,
        rust::Slice<float>  exceptions,
        rust::Slice<uint16_t> exceptions_positions,
        rust::Slice<int32_t> encoded_integers,
        float_state& stt) {
            std::unique_ptr<uint16_t> exc_c_arr (new uint16_t[alp::config::VECTOR_SIZE]);
            // call ALP's encode
            alp::encoder<float>::encode(
                input_vector.data(), 
                exceptions.data(),
                exceptions_positions.data(), 
                exc_c_arr.get(),
                encoded_integers.data(),
                stt
            );
            auto exception_count = exc_c_arr.get()[0];
            // std::cout << "Exceptions count: " << exception_count << std::endl;
            // for (size_t i = 1015; i < 1024; i++) {
            //     std::cout << "Encoded int: " << encoded_integers.data()[i] << std::endl;
            // } 
            return exception_count;
        }


    static inline int32_t alp_analyze_ffor(
        rust::Slice<const int32_t> input_vector, 
        uint8_t& bit_width) {
            std::unique_ptr<int32_t> ffor_base (new int32_t[alp::config::VECTOR_SIZE]);
            // call ALP's analyze_ffor
            alp::encoder<float>::analyze_ffor(
                input_vector.data(),
                bit_width,
                ffor_base.get()
            );
            auto base_arr = ffor_base.get();
            return base_arr[0];
    }


    static inline void fastlanes_ffor(
        rust::Slice<const int32_t> in,
        rust::Slice<int32_t> out,
        uint8_t bit_width, 
        rust::Slice<const int32_t> ffor_base) {
        // std::cout << sizeof(*out.data()) << std::endl;                
            // call Fastlane's unffor
            ffor::ffor(
                in.data(),
                out.data(),
                bit_width,
                ffor_base.data()
        );
        auto size = sizeof(out) - sizeof(out[0]);
        std::cout << "alp_state_values: Size of out: " << size << std::endl;
        // std::cout << sizeof(*out.data()) << std::endl;                
}

    // Checked
    static inline void alp_decode(
        rust::Slice<const int32_t> encoded_integers, 
        uint8_t fac_idx, 
        uint8_t exp_idx,
        rust::Slice<float> output) {
            // call ALP's decode
            alp::decoder<float>::decode(
                encoded_integers.data(),
                fac_idx,
                exp_idx,
                output.data()
            );
    }


    static inline void fastlanes_unffor(
        rust::Slice<const int32_t> in,
        rust::Slice<int32_t> out, 
        uint8_t bit_width, 
        rust::Slice<const int32_t> ffor_base) {
            // call Fastlane's unffor
            unffor::unffor(
                in.data(),
                out.data(),
                bit_width,
                ffor_base.data());

    }


    void alp_patch_exceptions(
        rust::Slice<float> output,
        rust::Slice<const float> exceptions,
        rust::Slice<const uint16_t> exceptions_positions,
        rust::Slice<const uint16_t> exceptions_count) {
            // call ALP's patch_exceptions
            alp::decoder<float>::patch_exceptions(
                output.data(),
                exceptions.data(),
                exceptions_positions.data(),
                exceptions_count.data());
    }


    int32_t get_state_scheme(const float_state& stt) {
        // checks for the passed state variable and returns the integer based on enum value for the ALP scheme
        // 1 - ALP; 2 - ALP_RD; -1 - Error 
        int32_t result = 0;
        switch (stt.scheme) {
        case alp::Scheme::ALP:
            result = 1;
            break;
        case alp::Scheme::ALP_RD:
            result = 2;
            break;
        case alp::Scheme::INVALID:
            result = 99;
            break;
        default:
            result = -1;
            break;
            }
        return result;
    }


    // returns ALP state factorial
    static uint8_t get_state_fac(const float_state& stt) {
        return stt.fac;
    }


    // returns ALP state exponent
    static uint8_t get_state_exp(const float_state& stt) {
        return stt.exp; 
    }
     // test methods
    int32_t new_state(){
        return 1;
    }


    size_t get_sampled_values_n(const float_state& stt) {
        return stt.sampled_values_n;
    };


    void change_sampled_values_n(float_state& stt) {
        // test method to write into stt
        stt.sampled_values_n = 10;
    } 
    void add_one(rust::Slice<float> arr) {
        for (size_t i = 0; i < arr.size(); i++) {
            arr.data()[i]++;
        }
    }


} // namespace alp_utils