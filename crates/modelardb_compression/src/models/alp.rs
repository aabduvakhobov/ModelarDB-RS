use modelardb_types::types::{Timestamp, UnivariateId, UnivariateIdBuilder, Value, ValueBuilder};


use crate::models;
use crate::models::bits::{BitReader, BitVecBuilder};
use crate::models::ErrorBound;

use crate::error::{ModelarDbCompressionError, Result};


use arrow::util::bit_util::ceil;
use alp_sys::{VECTOR_SIZE, N_VECTORS_PER_ROWGROUP, ROWGROUP_SIZE};
use alp_sys::ffi;
use alp_sys::{get_rowgroup, get_rowgroup_vector};

// ALP's main components that need to be stored
    /// The number of bits used to encode the base value.
    // bit_width: u8,
    // /// The base value is the value that is added to the exceptions.
    // base_arr: i32,
    // /// Exceptions are the values that ALP could not compress.
    // exceptions: Vec<Value>,
    // /// The positions of the exceptions in the original data.
    // exceptions_positions: Vec<u16>,
    // /// Exponent
    // exponent: u8,
    // /// Factor
    // factor: u8,
    // /// Array received from Fastlanes FoR compression
    // ffor_arr: Vec<i32>

    // ALP's state is returned

pub struct ALP {
    /// Maximum relative error for the value of each data point.
    error_bound: ErrorBound,
    /// The minimum value in the data.
    min_value: Value,
    /// The maximum value in the data.
    max_value: Value,
    /// Values compressed using XOR and a variable length binary encoding.
    compressed_values: BitVecBuilder,
}

impl ALP {
    /// Create a new ALP model with the given error bound.
    pub fn new(error_bound: ErrorBound) -> Self {
        Self {
            error_bound,
            min_value: Value::MAX,
            max_value: Value::NAN,
            compressed_values: BitVecBuilder::new(),
        }
    }

    // Compresses all ALP artifacts into a bit vector and updates min and max values along the way
    pub fn compress_values(&mut self, values: &[Value], num_vectors_in_a_rowgroup: usize, stt: &mut ffi::float_state) -> Result<BitVecBuilder> {
        if !can_use_alp(stt) {
            return Err(ModelarDbCompressionError::InvalidArgument("ALP cannot be used".to_string()));
        }
        // Update min max values from the rowgroup
        self.update_min_max_value(&values);
        // Create a new bit vector 
        let compressed_values: BitVecBuilder = BitVecBuilder::new();
        // here Alp compresses the whole rowgroup       
        let mut exceptions: [f32; VECTOR_SIZE] = [f32::INFINITY; VECTOR_SIZE];
        // let mut exceptions_positions: Vec<u16> = Vec::with_capacity(VECTOR_SIZE);
        let mut exceptions_positions = [0u16; VECTOR_SIZE];
        // let mut exceptions_positions: [u16; VECTOR_SIZE] = [0; VECTOR_SIZE];
        let mut encoded_integers: [i32; VECTOR_SIZE] = [0; VECTOR_SIZE];
        let mut ffor_arr: [i32; VECTOR_SIZE] = [0; VECTOR_SIZE];
        let mut decoded_arr: [f32; VECTOR_SIZE] = [0f32; VECTOR_SIZE];
        let mut bit_width = 0u8;
        
        for vector_idx in 0..num_vectors_in_a_rowgroup {
            let mut ffor_arr: [i32; VECTOR_SIZE] = [0; VECTOR_SIZE];
            // let mut ffor_arr: Vec<i32> = Vec::with_capacity(VECTOR_SIZE); // one way to allocate memory
            let mut unffor_arr: Vec<i32> = Vec::with_capacity(VECTOR_SIZE);
            
            let exception_cnt = ffi::alp_encode(
                values,
                &mut exceptions, 
                &mut exceptions_positions, 
                &mut encoded_integers,
                stt.as_mut().unwrap()
            );
            println!("Main.rs: exception_count: {}", exception_cnt);

            let trimmed_exceptions_positions = &exceptions_positions[0..exception_cnt as usize];
            let trimmed_exceptions = &exceptions[0..exception_cnt as usize];

            let base_arr = ffi::alp_analyze_ffor(
                &encoded_integers, 
                &mut bit_width
            );
            // ffi::fastlanes_ffor(encoded_arr, ffor_arr, bit_width, base_arr);
            ffi::fastlanes_ffor(
                &encoded_integers, 
                &mut ffor_arr, 
                bit_width,
                &[base_arr]
            );
            println!("Main.rs: ffor_arr size: {}", mem::size_of_val(&ffor_arr));
            let new_ffor = &ffor_arr[0..bit_width as usize ];
            println!("Main.rs: new_ffor size: {}", mem::size_of_val(new_ffor));
            // write the compressed values to the bit vector
        }
        Ok(compressed_values)
    }
    

    /// Update the current minimum, maximum from the given rowgroup 
    fn update_min_max_value(&mut self, values: &[Value]) {
        self.min_value = Value::min(self.min_value, values.iter().min().unwrap());
        self.max_value = Value::max(self.max_value, values.iter().max().unwrap());
    }

    /// Return the values compressed using ALP with a variable length binary
    /// encoding, the compressed minimum value, and the compressed maximum value.
    pub fn model(self) -> (Vec<u8>, Value, Value) {
        (
            self.compressed_values.finish(),
            self.min_value,
            self.max_value,
        )
    }
}


pub fn init(cur_rg: &[f32], row_group_id: usize, num_values_per_rowgroup: usize) -> ffi::float_state {
    // ALP compression strategy:
    // 1. Split your array into RowGroups that consist of 100 * Vectors
    let mut stt  = ffi::new_float_state(); 
    ffi::alp_init(
        &cur_rg, 
        row_group_id, 
        num_values_per_rowgroup, 
        stt.as_mut().unwrap()
    );
    sst
}

// Returns true if ALP can be used
pub fn can_use_alp(stt: &float_state) -> bool {
    let scheme = ffi::get_state_scheme(stt.as_ref().unwrap());
    return scheme == 1;
}

pub fn get_rowgroup(rowgroup_id: usize, v_value_per_rowgroup: usize, rowgroups: &[f32] )-> &[f32] {
    return &rowgroups[rowgroup_id*v_value_per_rowgroup..(rowgroup_id+1)*v_value_per_rowgroup];
}

pub fn get_rowgroup_vector( vec_id: usize, rowgroup_id: usize, rowgroups: &[f32])->  &[f32] {
    let offset =  (rowgroup_id * N_VECTORS_PER_ROWGROUP + vec_id) * VECTOR_SIZE;
    return &rowgroups[offset..offset+VECTOR_SIZE];
}





