use core::f32;

use cxx::UniquePtr;
use modelardb_types::types::{Timestamp, UnivariateId, UnivariateIdBuilder, Value, ValueBuilder};


use crate::models::ErrorBound;

use arrow::util::bit_util::ceil;
use alp_cc::{VECTOR_SIZE, ROWGROUP_SIZE};
use alp_cc::ffi::{self, float_state};

pub const MAGIC_NUMBER: u16 = 1025;

pub struct ALP {
    /// Maximum relative error for the value of each data point.
    error_bound: ErrorBound,
    /// The minimum value in the data.
    min_value: Value,
    /// The maximum value in the data.
    max_value: Value,
    /// Values compressed using XOR and a variable length binary encoding.
    compressed_values: Vec<u8>,
}

impl ALP {
    /// Create a new ALP model with the given error bound.
    pub fn new(error_bound: ErrorBound) -> Self {
        Self {
            error_bound,
            min_value: Value::MAX,
            max_value: Value::NAN,
            compressed_values: Vec::with_capacity(VECTOR_SIZE),
        }
    }

    // Compresses all ALP artifacts into a bit vector and updates min and max values along the way
    pub fn compress_values(&mut self, values: &[Value], num_vectors_in_a_rowgroup: usize, is_padded: bool, stt: &mut UniquePtr<float_state>) {
        if !can_use_alp(&stt) {
            panic!("ALP cannot be used");
        }
        // Update min max values from the rowgroup
        self.update_min_max_value(&values);
        // Create a new bit vector
        for vector_idx in 0..num_vectors_in_a_rowgroup {
            // Assign value containers
            let mut exceptions = [f32::INFINITY; VECTOR_SIZE];
            let mut exceptions_positions = [0u16; VECTOR_SIZE];
            let mut encoded_integers = [0; VECTOR_SIZE];
            let mut ffor_array= [0; VECTOR_SIZE];                  
            let mut bit_width = 0u8;
            // get vector data
            let data_p = get_rowgroup_vector(vector_idx, &values);     
            let exception_cnt = ffi::alp_encode(
                data_p,
                &mut exceptions, 
                &mut exceptions_positions, 
                &mut encoded_integers,
                stt.as_mut().unwrap()
            );
            let base_arr = ffi::alp_analyze_ffor(
                &encoded_integers, 
                &mut bit_width
            );

            let base_arr_slice = [base_arr];
            // println!("Main.rs: ffor_arr size: {}", mem::size_of_val(&ffor_arr));
            ffi::fastlanes_ffor(
                &encoded_integers, 
                &mut ffor_array, 
                bit_width,
                &base_arr_slice
            );
            // pass full ffor_arr container and only store values using the formula
            // write everything to bitvector
            // ffor_count can be deducted from updated bit_width value
            let ffor_count = 128 * bit_width as usize / std::mem::size_of::<i32>();
            // write factor, exponent, bit_width and base_array to byte vector
            let factor = ffi::get_state_fac(&stt);
            let exponent = ffi::get_state_exp(&stt);
            self.write_metadata(
                factor,
                exponent, 
                bit_width, 
                base_arr
            );
            let do_padding = if is_padded && vector_idx == num_vectors_in_a_rowgroup-1 {true} else {false};
            self.write_exceptions(
                exception_cnt,
                &exceptions, 
                &exceptions_positions, 
                do_padding,
            );

            self.write_encoded_integers(
                ffor_count as u16,
                &ffor_array,
            );
        }
    }
    

    fn write_metadata(&mut self, factor: u8, exponent: u8, bit_width: u8, base_arr: i32) {
            self.compressed_values.push(factor);
            self.compressed_values.push(exponent);
            self.compressed_values.push(bit_width);
            self.compressed_values.extend_from_slice(&base_arr.to_le_bytes());
    }


    fn write_exceptions(&mut self, exception_cnt: u16, exceptions: &[f32], exceptions_positions: &[u16], do_padding: bool) {
        // First write the exception count in 11 bits since max value can be 1024
        if do_padding {
            self.compressed_values.extend_from_slice(&(exception_cnt + MAGIC_NUMBER).to_le_bytes());
        } else {
            self.compressed_values.extend_from_slice(&exception_cnt.to_le_bytes());
        }
        // compressed_values.append_bits(exception_cnt as u64, 11);
            if exception_cnt != 0 {
                let offset = exception_cnt as usize;
                // Then write the exceptions in 32bits each
                self.compressed_values.extend_from_slice(to_u8_slice::<f32>(&exceptions[..offset]));
                self.compressed_values.extend_from_slice(to_u8_slice::<u16>(&exceptions_positions[..offset]));
            }
    }


    fn write_encoded_integers(&mut self, ffor_count: u16, ffor_arr: &[i32]) {
        // Write the ffor count in 8 bits
        self.compressed_values.extend_from_slice(&ffor_count.to_le_bytes()); 
        if ffor_count != 0 {
            // Then write the ffor_arr in 32bits
            let offset = ffor_count as usize;
            self.compressed_values.extend_from_slice(to_u8_slice(&ffor_arr[..offset]));
        }
    }

    /// Update the current minimum, maximum from the given rowgroup 
    fn update_min_max_value(&mut self, values: &[Value]) {
        self.min_value = Value::min(self.min_value, values.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
        self.max_value = Value::max(self.max_value, values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    }

    /// Return the values compressed using ALP with a variable length binary
    /// encoding, the compressed minimum value, and the compressed maximum value.
    pub fn model(self) -> (Vec<u8>, Value, Value) {
        (
            self.compressed_values,
            self.min_value,
            self.max_value,
        )
    }
}


pub fn sum(length: usize, values: &[u8]) -> Value {
    // This function replicates code from alp::grid() as it isn't necessary
    // to store the univariate ids, timestamps, and values in arrays for a sum.
    // So any changes to the decompression must be mirrored in alp::grid().
    let mut sum: Value = 0.0;
    let mut read_bytes = 0;
    let num_vectors_in_a_rowgroup: usize = ceil(length, ROWGROUP_SIZE);
    for _ in 0..num_vectors_in_a_rowgroup {
        // now start decoding values from compressed array and write them to decompressed valuebuilder
        let (factor, exponent, bit_width, base_arr, bytes) = read_metadata(&values, read_bytes);
        read_bytes = bytes;
        // dbg!(read_bytes);
        let (exception_count, exceptions, exception_positions, bytes, is_padded) = read_exceptions(&values, read_bytes);
        read_bytes = bytes;
        // dbg!(read_bytes);
        let (ffor_count, ffor_arr, bytes) = read_encoded_integers(&values, read_bytes);
        read_bytes = bytes;
        // dbg!(read_bytes);
        let mut out = [0; 1024];
        let mut decoded_floats = [0f32; 1024];

        ffi::fastlanes_unffor(
            &ffor_arr[..ffor_count as usize],
            &mut out,
            bit_width, 
            &[base_arr],
        );

        ffi::alp_decode(
            &out, 
            factor, 
            exponent, 
            &mut decoded_floats
        );

        ffi::alp_patch_exceptions(
            &mut decoded_floats, 
            &exceptions, 
            &exception_positions, 
            &[exception_count]
        );

        if is_padded {
            let padded_value = decoded_floats.last().unwrap().clone();
            if padded_value != 0.0 {
                let mut reverse_arr = decoded_floats.clone();
                reverse_arr.reverse();
                // safe to unwrap since it is controlled
                let padded_values = decoded_floats.len() - reverse_arr.iter().position(|&x| x!= padded_value ).unwrap();
                for i in padded_values..decoded_floats.len() {
                    decoded_floats[i] = 0.0;
                }
                // let decoded_floats = &decoded_floats[..(decoded_floats.len() - padded_values)];
            }
        }
        // sum decoded array
        sum += decoded_floats.iter().sum::<f32>();
    }
    sum
}

pub fn grid(
    univariate_id: UnivariateId,
    values: &[u8],
    univariate_id_builder: &mut UnivariateIdBuilder,
    timestamps: &[Timestamp],
    value_builder: &mut ValueBuilder,
) {
    univariate_id_builder.append_value(univariate_id);
    let mut read_bytes = 0;
    let num_vectors_in_a_rowgroup: usize = ceil(timestamps.len(), ROWGROUP_SIZE);
    for _ in 0..num_vectors_in_a_rowgroup {
        // now start decoding values from compressed array and write them to decompressed valuebuilder
        let (factor, exponent, bit_width, base_arr, bytes) = read_metadata(&values, read_bytes);
        read_bytes = bytes;
        // dbg!(read_bytes);
        let (exception_count, exceptions, exception_positions, bytes, is_padded) = read_exceptions(&values, read_bytes);
        read_bytes = bytes;
        // dbg!(read_bytes);
        let (ffor_count, ffor_arr, bytes) = read_encoded_integers(&values, read_bytes);
        read_bytes = bytes;
        // dbg!(read_bytes);
        let mut out = [0; 1024];
        let mut decoded_floats = [0f32; 1024];

        ffi::fastlanes_unffor(
            &ffor_arr[..ffor_count as usize],
            &mut out,
            bit_width, 
            &[base_arr],
        );

        ffi::alp_decode(
            &out, 
            factor, 
            exponent, 
            &mut decoded_floats
        );

        ffi::alp_patch_exceptions(
            &mut decoded_floats, 
            &exceptions, 
            &exception_positions, 
            &[exception_count]
        );

        if is_padded {
            let padded_value = decoded_floats.last().unwrap().clone();
            let mut reverse_arr = decoded_floats.clone();
            reverse_arr.reverse();
            // safe to unwrap since it is controlled
            let padded_values = reverse_arr.iter().position(|&x| x!= padded_value ).unwrap();
            value_builder.append_slice(&decoded_floats[..(decoded_floats.len() - padded_values)]);
        } else {
            value_builder.append_slice(&decoded_floats);
        }
    }
}

pub fn init(current_rowgroup: &[f32], row_group_id: usize, num_values_per_rowgroup: usize) -> cxx::UniquePtr<ffi::float_state> {
    // ALP compression strategy:
    // 1. Split your array into RowGroups that consist of 100 * Vectors
    let mut stt  = ffi::new_float_state(); 
    ffi::alp_init(
        &current_rowgroup, 
        row_group_id, 
        num_values_per_rowgroup, 
        stt.as_mut().unwrap()
    );
    stt
}

// Returns true if ALP can be used
pub fn can_use_alp(stt: &float_state) -> bool {
    let scheme = ffi::get_state_scheme(&stt);
    return scheme == 1;
}

pub fn get_rowgroup(rowgroup_id: usize, rowgroups: &[Value] )-> &[f32] {
    let mut upper_bound = (rowgroup_id+1)*ROWGROUP_SIZE;
    upper_bound = if upper_bound <= rowgroups.len() { upper_bound } else { rowgroups.len() };
    let values = &rowgroups[rowgroup_id*ROWGROUP_SIZE..upper_bound];
    values // slice(rowgroup_id, 1024);
}

pub fn perform_padding(current_rowgroup: &[f32], is_padded: &mut bool) -> Vec<f32> {
    let num_values_to_be_padded = if current_rowgroup.len() % VECTOR_SIZE != 0 {
        *is_padded = true;
        VECTOR_SIZE - (current_rowgroup.len() % VECTOR_SIZE)
    } else {
        0
    };
    let value_to_be_padded = if *current_rowgroup.last().unwrap() != 0.0 { 0.0 } else { 1.0 };
    let mut padded_vals = vec![value_to_be_padded; num_values_to_be_padded];

    let mut final_vec = current_rowgroup.to_vec();
    final_vec.append(&mut padded_vals);
    final_vec
}

fn get_rowgroup_vector(vector_id: usize, rowgroup: &[Value])->  &[f32] {
    return &rowgroup[vector_id*VECTOR_SIZE..(vector_id+1) * VECTOR_SIZE];
}

// TODO: highly unsafe function, generic type must be restricted  
fn to_u8_slice<T>(t_slice: &[T]) -> &[u8] {
    let (_, u8_slice, _) = unsafe { t_slice.align_to::<u8>() };    
    u8_slice
}

fn read_metadata(bytes: &[u8], read_byte_index: usize) -> (u8, u8, u8, i32, usize) {
    let mut more_bytes_readed = read_byte_index;
    let factor = bytes[more_bytes_readed];
    more_bytes_readed += 1;
    let exponent = bytes[more_bytes_readed];
    more_bytes_readed += 1;
    let bit_width = bytes[more_bytes_readed];
    let base_arr = i32::from_le_bytes(bytes[more_bytes_readed+1..more_bytes_readed+5].try_into().unwrap());// from_u8_slice(bytes[read_bytes+4..read_bytes+8]) ;
    more_bytes_readed += 5;
    (factor, exponent, bit_width, base_arr, more_bytes_readed)
}

fn read_exceptions(bytes: &[u8], read_byte_index: usize) -> (u16, [f32; 1024], [u16; 1024], usize, bool) {
    let mut more_bytes_readed = read_byte_index;
    let mut exceptions_count = u16::from_le_bytes(bytes[more_bytes_readed..more_bytes_readed+2].try_into().unwrap());
    let mut is_padded = false;
    if exceptions_count >= MAGIC_NUMBER {
        exceptions_count -= MAGIC_NUMBER;
        is_padded = true;
    }

    more_bytes_readed += 2;
    let mut exceptions = [0.0; 1024];
    let mut exception_positions = [0u16; 1024];
    if exceptions_count != 0 {
        for i in 0..exceptions_count as usize {
            exceptions[i] = f32::from_le_bytes(bytes[more_bytes_readed..more_bytes_readed+4].try_into().unwrap()) ;
            more_bytes_readed += 4;
        }
        // let exceptions: &[f32] = from_u8_slice(&bytes[more_bytes_readed..more_bytes_readed + (mem::size_of::<f32>() * exceptions_count as usize)]);
        // more_bytes_readed += 4 * exceptions_count as usize;
        for i in 0..exceptions_count as usize {
            exception_positions[i] = u16::from_le_bytes(bytes[more_bytes_readed..more_bytes_readed+2].try_into().unwrap()) ;
            more_bytes_readed += 2;
        }
        return (exceptions_count, exceptions, exception_positions, more_bytes_readed, is_padded)
    }
    // else we return error type some random bytes and return them without incrementing the counter
    (exceptions_count, exceptions, exception_positions, more_bytes_readed, is_padded)
}

fn read_encoded_integers(bytes: &[u8], read_byte_index: usize) -> (u16, [i32;1024], usize) {
    let mut more_bytes_readed = read_byte_index;
    let mut ffor_arr = [0; 1024];
    let ffor_count = u16::from_le_bytes(bytes[more_bytes_readed..more_bytes_readed+2].try_into().unwrap());
    more_bytes_readed += 2;
    if ffor_count != 0 {
        for i in 0..ffor_count as usize {
            ffor_arr[i] = i32::from_le_bytes(bytes[more_bytes_readed..more_bytes_readed+4 ].try_into().unwrap() );
            more_bytes_readed += 4;
        }
        // let ffor_arr: &[i32] = from_u8_slice(&bytes[more_bytes_readed..more_bytes_readed + (std::mem::size_of::<i32>() * ffor_count as usize)]);
        // more_bytes_readed += std::mem::size_of::<i32>() * ffor_count as usize;
        return (ffor_count, ffor_arr, more_bytes_readed)
    }
    (ffor_count, ffor_arr, more_bytes_readed)
}


