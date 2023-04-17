/* Copyright 2022 The ModelarDB Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Compress batches of sorted data points represented by a [`TimestampArray`] and a [`ValueArray`]
//! using the model types in [`models`] to produce compressed segments.

pub mod models;

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    BinaryArray, BinaryBuilder, Float32Array, Float32Builder, UInt64Array, UInt64Builder,
    UInt8Array, UInt8Builder,
};
use arrow::record_batch::RecordBatch;
use modelardb_common::errors::ModelarDbError;
use modelardb_common::schemas::COMPRESSED_SCHEMA;
use modelardb_common::types::{
    Timestamp, TimestampArray, TimestampBuilder, Value, ValueArray, ValueBuilder,
};

use crate::models::{pmc_mean::PMCMean, swing::Swing, timestamps, ErrorBound, SelectedModel};

/// Compress `uncompressed_timestamps` using a start time, end time, and a sampling interval if
/// regular and delta-of-deltas followed by a variable length binary encoding if irregular.
/// `uncompressed_values` is compressed within `error_bound` using the model types in [`models`].
/// Assumes `uncompressed_timestamps` and `uncompressed_values` are sorted according to
/// `uncompressed_timestamps`. Returns [`CompressionError`](ModelarDbError::CompressionError) if
/// `uncompressed_timestamps` and `uncompressed_values` have different lengths, otherwise the
/// resulting compressed segments are returned as a [`RecordBatch`] with the [`COMPRESSED_SCHEMA`]
/// schema.
pub fn try_compress(
    univariate_id: u64,
    uncompressed_timestamps: &TimestampArray,
    uncompressed_values: &ValueArray,
    error_bound: ErrorBound,
) -> Result<RecordBatch, ModelarDbError> {
    // The uncompressed data must be passed as arrays instead of a RecordBatch as a TimestampArray
    // and a ValueArray is the only supported input. However, as a result it is necessary to verify
    // they have the same length.
    if uncompressed_timestamps.len() != uncompressed_values.len() {
        return Err(ModelarDbError::CompressionError(
            "Uncompressed timestamps and uncompressed values have different lengths.".to_owned(),
        ));
    }

    // Enough memory for end_index compressed segments are allocated to never require reallocation
    // as one compressed segment is created per data point in the absolute worst case.
    let end_index = uncompressed_timestamps.len();
    let mut compressed_record_batch_builder = CompressedSegmentBatchBuilder::new(end_index);

    // Compress the uncompressed timestamps and uncompressed values.
    let mut current_index = 0;
    let mut residual_start_index = 0;
    while current_index < end_index {
        // Propose a model that represents the values from current_index to index within error_bound
        // where index <= end_index.
        let compressed_model_builder = CompressedModelBuilder::new(
            current_index,
            end_index,
            uncompressed_timestamps,
            uncompressed_values,
            error_bound,
        );

        let selected_model = compressed_model_builder.finish();

        // The selected model is only stored as part of a compressed segment if it uses less storage
        // space per value than the uncompressed values it represents.
        if selected_model.bytes_per_value <= models::VALUE_SIZE_IN_BYTES as f32 {
            // Compress residual data points with Gorilla and store them in a compressed segment.
            if residual_start_index != current_index {
                compress_and_store_residual_value_range(
                    univariate_id,
                    residual_start_index,
                    current_index - 1,
                    uncompressed_timestamps,
                    uncompressed_values,
                    &mut compressed_record_batch_builder,
                )
            }

            // Store the selected model and the corresponding timestamps in a compressed segment.
            store_selected_model_in_batch_builder(
                univariate_id,
                current_index,
                &selected_model,
                uncompressed_timestamps,
                &mut compressed_record_batch_builder,
            );

            // Update the indices to specify that all data points until now has been compressed.
            current_index = selected_model.end_index + 1;
            residual_start_index = current_index;
        } else {
            // The potentially lossy models could not efficiently encode the sub-sequence starting
            // at current_index, the residual values will instead be compressed using Gorilla.
            current_index += 1;
        }
    }

    // Compress the last residual data points with Gorilla and store them in a compressed segment.
    if residual_start_index != current_index {
        compress_and_store_residual_value_range(
            univariate_id,
            residual_start_index,
            current_index - 1,
            uncompressed_timestamps,
            uncompressed_values,
            &mut compressed_record_batch_builder,
        )
    }

    Ok(compressed_record_batch_builder.finish())
}

/// For the time series with `univariate_id`, compress the values from `start_index` to `end_index`
/// in `uncompressed_values` using [`Gorilla`] and store the resulting model with the corresponding
/// timestamps from `uncompressed_timestamps` as a segment in `compressed_record_batch_builder`.
fn compress_and_store_residual_value_range(
    univariate_id: u64,
    start_index: usize,
    end_index: usize,
    uncompressed_timestamps: &TimestampArray,
    uncompressed_values: &ValueArray,
    compressed_record_batch_builder: &mut CompressedSegmentBatchBuilder,
) {
    let selected_model =
        models::compress_residual_value_range(start_index, end_index, uncompressed_values);

    store_selected_model_in_batch_builder(
        univariate_id,
        start_index,
        &selected_model,
        uncompressed_timestamps,
        compressed_record_batch_builder,
    );
}

/// Store the `selected_model` which represents values from the time series with `univariate_id`
/// from `start_index to `end_index` as part of a segment in `compressed_record_batch_builder`.
fn store_selected_model_in_batch_builder(
    univariate_id: u64,
    start_index: usize,
    selected_model: &SelectedModel,
    uncompressed_timestamps: &TimestampArray,
    compressed_record_batch_builder: &mut CompressedSegmentBatchBuilder,
) {
    // Add timestamps and error.
    let end_index = selected_model.end_index;
    let start_time = uncompressed_timestamps.value(start_index);
    let end_time = uncompressed_timestamps.value(end_index);
    let timestamps = timestamps::compress_residual_timestamps(
        &uncompressed_timestamps.values()[start_index..=end_index],
    );
    let error = f32::NAN; // TODO: compute and store the actual error.

    compressed_record_batch_builder.append_compressed_segment(
        univariate_id,
        selected_model.model_type_id,
        start_time,
        end_time,
        &timestamps,
        selected_model.min_value,
        selected_model.max_value,
        &selected_model.values,
        error,
    );
}

/// Merge segments in `compressed_segments` that:
/// * Are from same time series.
/// * Contain the exact same models.
/// * Are consecutive in terms of time.
/// Assumes that if the consecutive segments A, B, and C exist for a time series and the segments A
/// and C are in `compressed_segments` then B is also in `compressed_segments`. If only A and C are
/// in `compressed_segments` a segment that overlaps with B will be created if A and C are merged.
pub fn merge_segments(compressed_segments: RecordBatch) -> RecordBatch {
    // Extract the columns from the RecordBatch.
    modelardb_common::arrays!(
        compressed_segments,
        univariate_ids,
        model_type_ids,
        start_times,
        end_times,
        timestamps,
        min_values,
        max_values,
        values,
        errors
    );

    // For each segment, check if it can be merged with another adjacent segment.
    let num_rows = compressed_segments.num_rows();
    let mut can_segments_be_merged = false;
    let mut univariate_id_to_previous_index = HashMap::new();
    let mut indices_to_merge_per_univariate_id = HashMap::new();

    for current_index in 0..num_rows {
        let univariate_id = univariate_ids.value(current_index);
        let previous_index = *univariate_id_to_previous_index
            .get(&univariate_id)
            .unwrap_or(&current_index);

        if can_models_be_merged(
            previous_index,
            current_index,
            univariate_ids,
            model_type_ids,
            min_values,
            max_values,
            values,
        ) {
            indices_to_merge_per_univariate_id
                .entry(univariate_id)
                .or_insert_with(Vec::new)
                .push(Some(current_index));

            can_segments_be_merged = previous_index != current_index;
        } else {
            // unwrap() is safe as a segment is guaranteed to match itself.
            let indices_to_merge = indices_to_merge_per_univariate_id
                .get_mut(&univariate_id)
                .unwrap();
            indices_to_merge.push(None);
            indices_to_merge.push(Some(current_index));
        }

        univariate_id_to_previous_index.insert(univariate_id, current_index);
    }

    // If none of the segments can be merged return the original compressed
    // segments, otherwise return the smaller set of merged compressed segments.
    if can_segments_be_merged {
        let mut merged_compressed_segments = CompressedSegmentBatchBuilder::new(num_rows);
        let mut index_of_last_segment = 0;

        let mut timestamp_builder = TimestampBuilder::new();
        for (_, mut indices_to_merge) in indices_to_merge_per_univariate_id {
            indices_to_merge.push(None);
            for maybe_index in indices_to_merge {
                if let Some(index) = maybe_index {
                    // Merge timestamps.
                    let start_time = start_times.value(index);
                    let end_time = end_times.value(index);
                    let timestamps = timestamps.value(index);

                    timestamps::decompress_all_timestamps(
                        start_time,
                        end_time,
                        timestamps,
                        &mut timestamp_builder,
                    );

                    index_of_last_segment = index;
                } else {
                    let timestamps = timestamp_builder.finish();
                    let compressed_timestamps =
                        timestamps::compress_residual_timestamps(timestamps.values());

                    // Merge segments. The last segment's model is used for the merged
                    // segment as all of the segments contain the exact same model.
                    merged_compressed_segments.append_compressed_segment(
                        univariate_ids.value(index_of_last_segment),
                        model_type_ids.value(index_of_last_segment),
                        timestamps.value(0),
                        timestamps.value(timestamps.len() - 1),
                        &compressed_timestamps,
                        min_values.value(index_of_last_segment),
                        max_values.value(index_of_last_segment),
                        values.value(index_of_last_segment),
                        errors.value(index_of_last_segment),
                    );
                }
            }
        }
        merged_compressed_segments.finish()
    } else {
        compressed_segments
    }
}

/// Return [`true`] if the models at `previous_index` and `current_index` represent values from the
/// same time series, are of the same type, and are equivalent, otherwise [`false`]. Assumes the
/// arrays are the same length and that `previous_index` and `current_index` only access values in
/// the arrays.
fn can_models_be_merged(
    previous_index: usize,
    current_index: usize,
    univariate_ids: &UInt64Array,
    model_type_ids: &UInt8Array,
    min_values: &ValueArray,
    max_values: &ValueArray,
    values: &BinaryArray,
) -> bool {
    // f32 are converted to u32 with the same bitwise representation as f32
    // and f64 does not implement std::hash::Hash and thus cannot be hashed.
    (
        univariate_ids.value(previous_index),
        model_type_ids.value(previous_index),
        min_values.value(previous_index).to_bits(),
        max_values.value(previous_index).to_bits(),
        values.value(previous_index),
    ) == (
        univariate_ids.value(current_index),
        model_type_ids.value(current_index),
        min_values.value(current_index).to_bits(),
        max_values.value(current_index).to_bits(),
        values.value(current_index),
    )
}

/// A compressed model being built from an uncompressed segment using the potentially lossy model
/// types in [`models`]. Each of the potentially lossy model types is used to fit models to the data
/// points, and then the model that uses the fewest number of bytes per value is selected.
struct CompressedModelBuilder {
    /// Index of the first data point in `uncompressed_timestamps` and `uncompressed_values` the
    /// compressed model represents values for.
    start_index: usize,
    /// Constant function that currently represents the values in `uncompressed_values` from
    /// `start_index` to `start_index` + `pmc_mean.length`.
    pmc_mean: PMCMean,
    /// Indicates if `pmc_mean` could represent all values in `uncompressed_values` from
    /// `start_index` to `current_index` in `new()`.
    pmc_mean_could_fit_all: bool,
    /// Linear function that represents the values in `uncompressed_values` from `start_index` to
    /// `start_index` + `swing.length`.
    swing: Swing,
    /// Indicates if `swing` could represent all values in `uncompressed_values` from `start_index`
    /// to `current_index` in `new()`.
    swing_could_fit_all: bool,
}

impl CompressedModelBuilder {
    /// Create a compressed model that represents the values in `uncompressed_values` from
    /// `start_index` to index within `error_bound` where index <= `end_index`.
    fn new(
        start_index: usize,
        end_index: usize,
        uncompressed_timestamps: &TimestampArray,
        uncompressed_values: &ValueArray,
        error_bound: ErrorBound,
    ) -> Self {
        let mut compressed_segment_builder = Self {
            start_index,
            pmc_mean: PMCMean::new(error_bound),
            pmc_mean_could_fit_all: true,
            swing: Swing::new(error_bound),
            swing_could_fit_all: true,
        };

        let mut current_index = start_index;
        while compressed_segment_builder.can_fit_more() && current_index < end_index {
            let timestamp = uncompressed_timestamps.value(current_index);
            let value = uncompressed_values.value(current_index);
            compressed_segment_builder.try_to_update_models(timestamp, value);
            current_index += 1;
        }
        compressed_segment_builder
    }

    /// Attempt to update the current models to also represent the `value` of
    /// the data point collected at `timestamp`.
    fn try_to_update_models(&mut self, timestamp: Timestamp, value: Value) {
        debug_assert!(
            self.can_fit_more(),
            "The current models cannot be fitted to additional data points."
        );

        self.pmc_mean_could_fit_all = self.pmc_mean_could_fit_all && self.pmc_mean.fit_value(value);

        self.swing_could_fit_all =
            self.swing_could_fit_all && self.swing.fit_data_point(timestamp, value);
    }

    /// Return [`true`] if any of the current models can represent additional
    /// values, otherwise [`false`].
    fn can_fit_more(&self) -> bool {
        self.pmc_mean_could_fit_all || self.swing_could_fit_all
    }

    /// Return the model that requires the fewest number of bytes per value.
    fn finish(self) -> SelectedModel {
        SelectedModel::new(self.start_index, self.pmc_mean, self.swing)
    }
}

/// A batch of compressed segments being built.
struct CompressedSegmentBatchBuilder {
    /// Univariate ids of each compressed segment in the batch.
    univariate_ids: UInt64Builder,
    /// Model type ids of each compressed segment in the batch.
    model_type_ids: UInt8Builder,
    /// First timestamp of each compressed segment in the batch.
    start_times: TimestampBuilder,
    /// Last timestamp of each compressed segment in the batch.
    end_times: TimestampBuilder,
    /// Data required in addition to `start_times` and `end_times` to
    /// reconstruct the timestamps of each compressed segment in the batch.
    timestamps: BinaryBuilder,
    /// Minimum value of each compressed segment in the batch.
    min_values: ValueBuilder,
    /// Maximum value of each compressed segment in the batch.
    max_values: ValueBuilder,
    /// Data required in addition to `min_value` and `max_value` to reconstruct
    /// the values of each compressed segment in the batch within an error
    /// bound.
    values: BinaryBuilder,
    /// Actual error of each compressed segment in the batch.
    error: Float32Builder,
}

impl CompressedSegmentBatchBuilder {
    fn new(capacity: usize) -> Self {
        Self {
            univariate_ids: UInt64Builder::with_capacity(capacity),
            model_type_ids: UInt8Builder::with_capacity(capacity),
            start_times: TimestampBuilder::with_capacity(capacity),
            end_times: TimestampBuilder::with_capacity(capacity),
            timestamps: BinaryBuilder::with_capacity(capacity, capacity),
            min_values: ValueBuilder::with_capacity(capacity),
            max_values: ValueBuilder::with_capacity(capacity),
            values: BinaryBuilder::with_capacity(capacity, capacity),
            error: Float32Builder::with_capacity(capacity),
        }
    }

    /// Append a compressed segment to the builder.
    #[allow(clippy::too_many_arguments)]
    fn append_compressed_segment(
        &mut self,
        univariate_id: u64,
        model_type_id: u8,
        start_time: Timestamp,
        end_time: Timestamp,
        timestamps: &[u8],
        min_value: Value,
        max_value: Value,
        values: &[u8],
        error: f32,
    ) {
        self.univariate_ids.append_value(univariate_id);
        self.model_type_ids.append_value(model_type_id);
        self.start_times.append_value(start_time);
        self.end_times.append_value(end_time);
        self.timestamps.append_value(timestamps);
        self.min_values.append_value(min_value);
        self.max_values.append_value(max_value);
        self.values.append_value(values);
        self.error.append_value(error);
    }

    /// Return [`RecordBatch`] of compressed segments and consume the builder.
    fn finish(mut self) -> RecordBatch {
        RecordBatch::try_new(
            COMPRESSED_SCHEMA.0.clone(),
            vec![
                Arc::new(self.univariate_ids.finish()),
                Arc::new(self.model_type_ids.finish()),
                Arc::new(self.start_times.finish()),
                Arc::new(self.end_times.finish()),
                Arc::new(self.timestamps.finish()),
                Arc::new(self.min_values.finish()),
                Arc::new(self.max_values.finish()),
                Arc::new(self.values.finish()),
                Arc::new(self.error.finish()),
            ],
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::UInt8Array;

    use crate::models;
    use crate::test_util::StructureOfValues;

    const ERROR_BOUND_ZERO: f32 = 0.0;
    const ERROR_BOUND_FIVE: f32 = 5.0;
    const TRY_COMPRESS_TEST_LENGTH: usize = 50;

    // Tests for try_compress().
    #[test]
    fn test_try_compress_empty_time_series() {
        let values = vec![];
        let timestamps = vec![];
        let (_, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_ZERO);

        assert_eq!(0, compressed_record_batch.num_rows())
    }

    #[test]
    fn test_try_compress_regular_constant_time_series() {
        let timestamps = test_util::generate_timestamps(TRY_COMPRESS_TEST_LENGTH, false);
        let values =
            test_util::generate_values(&timestamps, StructureOfValues::Constant, None, None);

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_ZERO);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::PMC_MEAN_ID],
        )
    }

    #[test]
    fn test_try_compress_irregular_constant_time_series() {
        let timestamps = test_util::generate_timestamps(TRY_COMPRESS_TEST_LENGTH, true);
        let values =
            test_util::generate_values(&timestamps, StructureOfValues::Constant, None, None);

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_ZERO);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::PMC_MEAN_ID],
        )
    }

    #[test]
    fn test_try_compress_regular_almost_constant_time_series() {
        let timestamps = test_util::generate_timestamps(TRY_COMPRESS_TEST_LENGTH, false);
        let values = test_util::generate_values(
            &timestamps,
            StructureOfValues::Random,
            Some(9.8),
            Some(10.2),
        );

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_FIVE);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::PMC_MEAN_ID],
        )
    }

    #[test]
    fn test_try_compress_irregular_almost_constant_time_series() {
        let timestamps = test_util::generate_timestamps(TRY_COMPRESS_TEST_LENGTH, true);
        let values = test_util::generate_values(
            &timestamps,
            StructureOfValues::Random,
            Some(9.8),
            Some(10.2),
        );

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_FIVE);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::PMC_MEAN_ID],
        )
    }

    #[test]
    fn test_try_compress_regular_linear_time_series() {
        let timestamps = test_util::generate_timestamps(TRY_COMPRESS_TEST_LENGTH, false);
        let values = test_util::generate_values(&timestamps, StructureOfValues::Linear, None, None);

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_ZERO);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::SWING_ID],
        )
    }

    #[test]
    fn test_try_compress_irregular_linear_time_series() {
        let timestamps = test_util::generate_timestamps(TRY_COMPRESS_TEST_LENGTH, true);
        let values = test_util::generate_values(&timestamps, StructureOfValues::Linear, None, None);

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_FIVE);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::SWING_ID],
        )
    }

    #[test]
    fn test_try_compress_regular_almost_linear_time_series() {
        let timestamps = test_util::generate_timestamps(TRY_COMPRESS_TEST_LENGTH, false);
        let values = test_util::generate_values(
            &timestamps,
            StructureOfValues::AlmostLinear,
            Some(9.8),
            Some(10.2),
        );

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_FIVE);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::SWING_ID],
        )
    }

    #[test]
    fn test_try_compress_irregular_almost_linear_time_series() {
        let timestamps = test_util::generate_timestamps(TRY_COMPRESS_TEST_LENGTH, true);
        let values = test_util::generate_values(
            &timestamps,
            StructureOfValues::AlmostLinear,
            Some(9.8),
            Some(10.2),
        );

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_FIVE);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::SWING_ID],
        )
    }

    #[test]
    fn test_try_compress_regular_random_time_series() {
        let timestamps = test_util::generate_timestamps(TRY_COMPRESS_TEST_LENGTH, false);
        let values = test_util::generate_values(
            &timestamps,
            StructureOfValues::Random,
            Some(0.0),
            Some(f32::MAX),
        );

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_ZERO);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::GORILLA_ID],
        )
    }

    #[test]
    fn test_try_compress_irregular_random_time_series() {
        let timestamps = test_util::generate_timestamps(TRY_COMPRESS_TEST_LENGTH, true);
        let values = test_util::generate_values(
            &timestamps,
            StructureOfValues::Random,
            Some(0.0),
            Some(f32::MAX),
        );

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_ZERO);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::GORILLA_ID],
        )
    }

    #[test]
    fn test_try_compress_regular_random_linear_constant_time_series() {
        let timestamps = test_util::generate_timestamps(3 * TRY_COMPRESS_TEST_LENGTH, false);
        let mut constant = test_util::generate_values(
            &timestamps[0..TRY_COMPRESS_TEST_LENGTH],
            StructureOfValues::Constant,
            None,
            None,
        );
        let mut linear = test_util::generate_values(
            &timestamps[TRY_COMPRESS_TEST_LENGTH..2 * TRY_COMPRESS_TEST_LENGTH],
            StructureOfValues::Linear,
            None,
            None,
        );
        let mut random = test_util::generate_values(
            &timestamps[2 * TRY_COMPRESS_TEST_LENGTH..],
            StructureOfValues::Random,
            Some(0.0),
            Some(f32::MAX),
        );
        let mut values = vec![];
        values.append(&mut random);
        values.append(&mut linear);
        values.append(&mut constant);

        let (uncompressed_timestamps, compressed_record_batch) =
            create_and_compress_time_series(&values, &timestamps, ERROR_BOUND_ZERO);

        assert_compressed_record_batch_with_segments_from_regular_time_series(
            &uncompressed_timestamps,
            &compressed_record_batch,
            &[models::GORILLA_ID, models::SWING_ID, models::PMC_MEAN_ID],
        )
    }

    fn create_uncompressed_time_series(
        timestamps: &[Timestamp],
        values: &[Value],
    ) -> (TimestampArray, ValueArray) {
        let mut timestamps_builder = TimestampBuilder::with_capacity(timestamps.len());
        timestamps_builder.append_slice(timestamps);
        let mut values_builder = ValueBuilder::with_capacity(values.len());
        values_builder.append_slice(values);
        (timestamps_builder.finish(), values_builder.finish())
    }

    fn create_and_compress_time_series(
        values: &[f32],
        timestamps: &[i64],
        error_bound: f32,
    ) -> (TimestampArray, RecordBatch) {
        let (uncompressed_timestamps, uncompressed_values) =
            create_uncompressed_time_series(timestamps, values);
        let error_bound = ErrorBound::try_new(error_bound).unwrap();
        let compressed_record_batch = try_compress(
            1,
            &uncompressed_timestamps,
            &uncompressed_values,
            error_bound,
        )
        .unwrap();
        (uncompressed_timestamps, compressed_record_batch)
    }

    fn assert_compressed_record_batch_with_segments_from_regular_time_series(
        uncompressed_timestamps: &TimestampArray,
        compressed_record_batch: &RecordBatch,
        expected_model_type_ids: &[u8],
    ) {
        assert_eq!(
            expected_model_type_ids.len(),
            compressed_record_batch.num_rows()
        );

        let mut total_compressed_length = 0;
        for (segment, expected_model_type_id) in expected_model_type_ids.iter().enumerate() {
            let model_type_id =
                modelardb_common::array!(compressed_record_batch, 1, UInt8Array).value(segment);
            assert_eq!(*expected_model_type_id, model_type_id);

            let start_time =
                modelardb_common::array!(compressed_record_batch, 2, TimestampArray).value(segment);
            let end_time =
                modelardb_common::array!(compressed_record_batch, 3, TimestampArray).value(segment);
            let timestamps =
                modelardb_common::array!(compressed_record_batch, 4, BinaryArray).value(segment);

            total_compressed_length += models::len(start_time, end_time, timestamps);
        }
        assert_eq!(uncompressed_timestamps.len(), total_compressed_length);
    }

    // Tests for merge_segments().
    #[test]
    fn test_merge_compressed_segments_empty_batch() {
        let merged_record_batch = merge_segments(CompressedSegmentBatchBuilder::new(0).finish());
        assert_eq!(0, merged_record_batch.num_rows())
    }

    #[test]
    fn test_merge_compressed_segments_batch() {
        // merge_segments() currently merge segments with equivalent models.
        let univariate_id = 1;
        let model_type_id = 1;
        let values = &[];
        let min_value = 5.0;
        let max_value = 5.0;

        // Add a mix of different segments that can be merged into two segments.
        let mut compressed_record_batch_builder = CompressedSegmentBatchBuilder::new(10);

        for start_time in (100..2100).step_by(400) {
            compressed_record_batch_builder.append_compressed_segment(
                univariate_id,
                model_type_id,
                start_time,
                start_time + 100,
                &[],
                min_value,
                max_value,
                values,
                0.0,
            );
        }

        for start_time in (2500..4500).step_by(400) {
            compressed_record_batch_builder.append_compressed_segment(
                univariate_id,
                model_type_id + 1,
                start_time + 200,
                start_time + 300,
                &[],
                -min_value,
                -max_value,
                values,
                10.0,
            );
        }

        let compressed_record_batch = compressed_record_batch_builder.finish();
        let merged_record_batch = merge_segments(compressed_record_batch);

        // Extract the columns from the RecordBatch.
        let start_times = modelardb_common::array!(merged_record_batch, 2, TimestampArray);
        let end_times = modelardb_common::array!(merged_record_batch, 3, TimestampArray);
        let timestamps = modelardb_common::array!(merged_record_batch, 4, BinaryArray);
        let min_values = modelardb_common::array!(merged_record_batch, 5, ValueArray);
        let max_values = modelardb_common::array!(merged_record_batch, 6, ValueArray);
        let values = modelardb_common::array!(merged_record_batch, 7, BinaryArray);
        let errors = modelardb_common::array!(merged_record_batch, 8, Float32Array);

        // Assert that the number of segments are correct.
        assert_eq!(2, merged_record_batch.num_rows());

        // Assert that the timestamps are correct.
        let mut decompressed_timestamps = TimestampBuilder::with_capacity(10);
        timestamps::decompress_all_timestamps(
            start_times.value(0),
            end_times.value(0),
            timestamps.value(0),
            &mut decompressed_timestamps,
        );
        assert_eq!(10, decompressed_timestamps.finish().len());

        timestamps::decompress_all_timestamps(
            start_times.value(1),
            end_times.value(1),
            timestamps.value(1),
            &mut decompressed_timestamps,
        );
        assert_eq!(10, decompressed_timestamps.finish().len());

        // Assert that the models are correct.
        let (positive, negative) = if start_times.value(0) == 100 {
            (0, 1)
        } else {
            (1, 0)
        };

        let value: &[u8] = &[];
        assert_eq!(value, values.value(positive));
        assert_eq!(min_value, min_values.value(positive));
        assert_eq!(max_value, max_values.value(positive));

        assert_eq!(value, values.value(negative));
        assert_eq!(-min_value, min_values.value(negative));
        assert_eq!(-max_value, max_values.value(negative));

        // Assert that the errors are correct.
        assert_eq!(0.0, errors.value(positive));
        assert_eq!(10.0, errors.value(negative));
    }

    // Tests for can_models_be_merged().
    #[test]
    fn test_models_can_be_merged() {
        assert!(can_models_be_merged(
            0,
            1,
            &UInt64Array::from_iter_values([1, 1]),
            &UInt8Array::from_iter_values([1, 1]),
            &ValueArray::from_iter_values([1.0, 1.0]),
            &ValueArray::from_iter_values([2.0, 2.0]),
            &BinaryArray::from_iter_values([[1], [1]])
        ))
    }

    #[test]
    fn test_models_with_different_univariate_ids_cannot_be_merged() {
        assert!(!can_models_be_merged(
            0,
            1,
            &UInt64Array::from_iter_values([1, 2]),
            &UInt8Array::from_iter_values([1, 1]),
            &ValueArray::from_iter_values([1.0, 1.0]),
            &ValueArray::from_iter_values([2.0, 2.0]),
            &BinaryArray::from_iter_values([[1], [1]])
        ))
    }

    #[test]
    fn test_models_with_different_model_types_cannot_be_merged() {
        assert!(!can_models_be_merged(
            0,
            1,
            &UInt64Array::from_iter_values([1, 1]),
            &UInt8Array::from_iter_values([1, 2]),
            &ValueArray::from_iter_values([1.0, 1.0]),
            &ValueArray::from_iter_values([2.0, 2.0]),
            &BinaryArray::from_iter_values([[1], [1]])
        ))
    }

    #[test]
    fn test_models_with_different_min_values_cannot_be_merged() {
        assert!(!can_models_be_merged(
            0,
            1,
            &UInt64Array::from_iter_values([1, 1]),
            &UInt8Array::from_iter_values([1, 1]),
            &ValueArray::from_iter_values([1.0, 2.0]),
            &ValueArray::from_iter_values([2.0, 2.0]),
            &BinaryArray::from_iter_values([[1], [1]])
        ))
    }

    #[test]
    fn test_models_with_different_max_values_cannot_be_merged() {
        assert!(!can_models_be_merged(
            0,
            1,
            &UInt64Array::from_iter_values([1, 1]),
            &UInt8Array::from_iter_values([1, 1]),
            &ValueArray::from_iter_values([1.0, 1.0]),
            &ValueArray::from_iter_values([2.0, 3.0]),
            &BinaryArray::from_iter_values([[1], [1]])
        ))
    }

    #[test]
    fn test_models_with_different_values_cannot_be_merged() {
        assert!(!can_models_be_merged(
            0,
            1,
            &UInt64Array::from_iter_values([1, 1]),
            &UInt8Array::from_iter_values([1, 1]),
            &ValueArray::from_iter_values([1.0, 1.0]),
            &ValueArray::from_iter_values([2.0, 2.0]),
            &BinaryArray::from_iter_values([[1], [2]])
        ))
    }
}

/// Separate module for utility functions.
#[cfg(test)]
pub mod test_util {
    use rand::distributions::Uniform;
    use rand::{thread_rng, Rng};

    pub enum StructureOfValues {
        Constant,
        Random,
        Linear,
        AlmostLinear,
    }

    /// Generate constant/random/linear/almost-linear test values with the
    /// [ThreadRng](rand::rngs::thread::ThreadRng) randomizer. The amount of values to be generated
    /// will match `timestamps` and their structure will match [`StructureOfValues`]. If `Random` is
    /// selected, `min` and `max` is the range of values which can be generated. If `AlmostLinear`
    /// is selected, `min` and `max` is the maximum and minimum change that should be applied from
    /// one value to the next. Returns the generated values as a [`Vec<f32>`].
    pub fn generate_values(
        timestamps: &[i64],
        data_type: StructureOfValues,
        min: Option<f32>,
        max: Option<f32>,
    ) -> Vec<f32> {
        let mut randomizer = thread_rng();
        match data_type {
            // Generates almost linear data.
            StructureOfValues::AlmostLinear => {
                // The variable a is regenerated if it is 0, to avoid generating constant data.
                let mut a: i64 = 0;
                while a == 0 {
                    a = thread_rng().gen_range(-10..10);
                }
                let b: i64 = thread_rng().gen_range(1..50);

                timestamps
                    .iter()
                    .map(|timestamp| {
                        (a * timestamp + b) as f32
                            + randomizer.sample(Uniform::from(min.unwrap()..max.unwrap()))
                    })
                    .collect()
            }
            // Generates linear data.
            StructureOfValues::Linear => {
                // The variable a is regenerated if it is 0, to avoid generating constant data.
                let mut a: i64 = 0;
                while a == 0 {
                    a = thread_rng().gen_range(-10..10);
                }
                let b: i64 = thread_rng().gen_range(1..50);

                timestamps
                    .iter()
                    .map(|timestamp| (a * timestamp + b) as f32)
                    .collect()
            }
            // Generates randomized data.
            StructureOfValues::Random => {
                let mut random = vec![];
                let mut randomizer = thread_rng();

                for _ in 0..timestamps.len() {
                    random.push(randomizer.sample(Uniform::from(min.unwrap()..max.unwrap())));
                }

                random
            }
            // Generates constant data.
            StructureOfValues::Constant => {
                vec![thread_rng().gen(); timestamps.len()]
            }
        }
    }

    /// Generate regular/irregular timestamps with the [ThreadRng](rand::rngs::thread::ThreadRng) randomizer.
    /// Select the length and type of timestamps to be generated using the parameters `length` and `irregular`.
    /// Returns the generated timestamps as a [`Vec`].
    pub fn generate_timestamps(length: usize, irregular: bool) -> Vec<i64> {
        let mut timestamps = vec![];
        if irregular {
            let mut randomizer = thread_rng();
            let mut previous_timestamp: i64 = 0;
            for _ in 0..length {
                let next_timestamp =
                    (randomizer.sample(Uniform::from(10..20))) + previous_timestamp;
                timestamps.push(next_timestamp);
                previous_timestamp = next_timestamp;
            }
        } else {
            timestamps = Vec::from_iter((100..(length + 1) as i64 * 100).step_by(100));
        }

        timestamps
    }
}
