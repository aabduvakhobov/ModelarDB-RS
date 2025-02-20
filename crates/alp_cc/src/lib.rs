use arrow::{array::PrimitiveArray, datatypes::Float32Type};

pub const VECTOR_SIZE: usize = 1024;
pub const N_VECTORS_PER_ROWGROUP: usize = 100;
pub const ROWGROUP_SIZE: usize = N_VECTORS_PER_ROWGROUP * 1024;


#[cxx::bridge(namespace="alp_utils")]
pub mod ffi{
    unsafe extern  "C++" {
        include!("alp_cc/include/alp_state_values.hpp");
        // write ALP C++ opaque types
        pub type float_state;


        fn new_float_state() -> UniquePtr<float_state>; // needs to return heap allocated raw pointer.
        // write ALP C++ function signatures to be used in the project
        // encoding
        fn alp_init(
            data_column: &[f32], 
            column_offset: usize,
            tuples_count: usize,
            stt: Pin<&mut float_state>);
        

        // encode function returns exceptions_count to read exceptions and its positions
        fn alp_encode(
            input_vector: &[f32],
            exceptions: &mut [f32],
            exceptions_positions: &mut [u16],
            encoded_integers: &mut [i32], // Encoded integers are always int64
            stt: Pin<&mut float_state>) -> u16;

        fn alp_analyze_ffor(
            input_vector: &[i32],
            bit_width: &mut u8            
        ) -> i32;


        fn fastlanes_ffor(
            n: &[i32], 
            out: &mut [i32],
            bit_width: u8, 
            ffor_base: &[i32]
        );

        // decoding
        fn fastlanes_unffor(
            n: &[i32],
            out: &mut [i32],
            bit_width: u8, 
            ffor_base: &[i32],
        );

        fn alp_decode(
            encoded_integers: &[i32], 
            fac_idx: u8, 
            exp_idx: u8, 
            output: &mut [f32],
        );


        fn alp_patch_exceptions(
            output: &mut [f32],
            exceptions: &[f32], 
            exceptions_positions: &[u16], 
            exceptions_count: &[u16],
        );
         
        fn get_state_scheme(stt: &float_state) -> i32;

        fn get_state_fac(stt: &float_state) -> u8;

        fn get_state_exp(stt: &float_state) -> u8;

        // test methods
        fn new_state() -> i32;

        fn add_one(arr: &mut [f32]);

        fn get_sampled_values_n(stt: &float_state) -> usize;

        fn change_sampled_values_n(stt: Pin<&mut float_state>);

    }
}