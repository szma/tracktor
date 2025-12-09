//! Multi-sensor LMB filters
//!
//! Filters that process measurements from multiple sensors and fuse
//! the results using various strategies.

use core::marker::PhantomData;
use nalgebra::RealField;
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::filter::{LabeledBirthModel, LmbFilterState};
use super::fusion::{
    ArithmeticAverageMerger, GeometricAverageMerger, IteratedCorrectorMerger, Merger,
    ParallelUpdateMerger,
};
use super::types::{LmbTrack, LmbTrackSet};
use crate::filters::phd::{Predicted, UpdateStats, Updated};
use crate::models::{ClutterModel, ObservationModel, TransitionModel};
use crate::types::labels::Label;
use crate::types::spaces::{Measurement, StateCovariance};

// ============================================================================
// Multi-Sensor LMB Filter
// ============================================================================

/// Multi-sensor LMB filter that fuses measurements from multiple sensors.
///
/// Each sensor has its own observation and clutter model. After prediction,
/// measurements from all sensors are processed and the results are fused
/// using the specified merger strategy.
#[cfg(feature = "alloc")]
#[derive(Clone)]
pub struct MultisensorLmbFilter<T, Trans, Birth, Mrg, const N: usize>
where
    T: RealField,
    Trans: TransitionModel<T, N>,
    Birth: LabeledBirthModel<T, N>,
    Mrg: Clone,
{
    /// Transition model (shared across sensors)
    pub transition: Trans,
    /// Birth model (shared)
    pub birth: Birth,
    /// Fusion strategy
    pub merger: Mrg,
    /// Sensor weights (for fusion)
    pub sensor_weights: Vec<T>,
    _marker: PhantomData<T>,
}

/// Configuration for a single sensor.
#[cfg(feature = "alloc")]
#[derive(Clone)]
pub struct SensorConfig<Obs, Clutter> {
    /// Observation model for this sensor
    pub observation: Obs,
    /// Clutter model for this sensor
    pub clutter: Clutter,
    /// Relative weight for fusion
    pub weight: f64,
}

#[cfg(feature = "alloc")]
impl<Obs, Clutter> SensorConfig<Obs, Clutter> {
    /// Creates a new sensor configuration.
    pub fn new(observation: Obs, clutter: Clutter, weight: f64) -> Self {
        Self {
            observation,
            clutter,
            weight,
        }
    }
}

#[cfg(feature = "alloc")]
impl<T, Trans, Birth, Mrg, const N: usize> MultisensorLmbFilter<T, Trans, Birth, Mrg, N>
where
    T: RealField + Float + Copy,
    Trans: TransitionModel<T, N> + Clone,
    Birth: LabeledBirthModel<T, N> + Clone,
    Mrg: Merger<T, N> + Clone,
{
    /// Creates a new multi-sensor LMB filter.
    pub fn new(transition: Trans, birth: Birth, merger: Mrg, num_sensors: usize) -> Self {
        let uniform_weight = T::one() / T::from(num_sensors).unwrap();
        Self {
            transition,
            birth,
            merger,
            sensor_weights: vec![uniform_weight; num_sensors],
            _marker: PhantomData,
        }
    }

    /// Creates a filter with custom sensor weights.
    pub fn with_weights(
        transition: Trans,
        birth: Birth,
        merger: Mrg,
        sensor_weights: Vec<T>,
    ) -> Self {
        Self {
            transition,
            birth,
            merger,
            sensor_weights,
            _marker: PhantomData,
        }
    }

    /// Creates an initial filter state.
    pub fn initial_state(&self) -> LmbFilterState<T, N, Updated> {
        LmbFilterState::new()
    }

    /// Runs one complete multi-sensor predict-update-fuse cycle.
    ///
    /// # Arguments
    ///
    /// * `state` - Current filter state
    /// * `sensor_measurements` - Measurements from each sensor
    /// * `sensor_configs` - Configuration for each sensor (observation and clutter models)
    /// * `dt` - Time step
    pub fn step<const M: usize, Obs, Clutter>(
        &self,
        state: LmbFilterState<T, N, Updated>,
        sensor_measurements: &[Vec<Measurement<T, M>>],
        sensor_configs: &[SensorConfig<Obs, Clutter>],
        dt: T,
    ) -> (LmbFilterState<T, N, Updated>, UpdateStats)
    where
        Obs: ObservationModel<T, N, M> + Clone,
        Clutter: ClutterModel<T, M> + Clone,
    {
        assert_eq!(
            sensor_measurements.len(),
            sensor_configs.len(),
            "Number of measurement sets must match number of sensor configs"
        );

        let num_sensors = sensor_measurements.len();
        if num_sensors == 0 {
            return (state, UpdateStats::default());
        }

        // Predict
        let predicted = state.predict(&self.transition, &self.birth, dt);

        // Process each sensor independently
        let mut sensor_states: Vec<LmbFilterState<T, N, Updated>> = Vec::with_capacity(num_sensors);
        let mut combined_stats = UpdateStats::default();

        for (_i, (measurements, config)) in sensor_measurements
            .iter()
            .zip(sensor_configs.iter())
            .enumerate()
        {
            // Clone predicted state for this sensor
            let sensor_predicted = LmbFilterState::<T, N, Predicted>::from_components(
                predicted.tracks.clone(),
                predicted.label_gen.clone(),
                predicted.time_step,
            );

            let (updated, stats) =
                sensor_predicted.update(measurements, &config.observation, &config.clutter);

            combined_stats.singular_covariance_count += stats.singular_covariance_count;
            combined_stats.zero_likelihood_count += stats.zero_likelihood_count;

            sensor_states.push(updated);
        }

        // Fuse sensor results
        let fused_tracks = fuse_sensor_tracks(&sensor_states, &self.sensor_weights, &self.merger);

        let fused_state = LmbFilterState::<T, N, Updated>::from_components(
            fused_tracks,
            predicted.label_gen,
            predicted.time_step,
        );

        (fused_state, combined_stats)
    }
}

/// Fuses tracks from multiple sensor updates.
#[cfg(feature = "alloc")]
fn fuse_sensor_tracks<T: RealField + Float + Copy, const N: usize, M: Merger<T, N>>(
    sensor_states: &[LmbFilterState<T, N, Updated>],
    sensor_weights: &[T],
    merger: &M,
) -> LmbTrackSet<T, N> {
    use alloc::collections::BTreeMap;

    if sensor_states.is_empty() {
        return LmbTrackSet::new();
    }

    // Collect all unique labels
    let mut tracks_by_label: BTreeMap<Label, Vec<(usize, &LmbTrack<T, N>)>> = BTreeMap::new();

    for (sensor_idx, state) in sensor_states.iter().enumerate() {
        for track in state.tracks.iter() {
            tracks_by_label
                .entry(track.label)
                .or_default()
                .push((sensor_idx, track));
        }
    }

    // Fuse tracks for each label
    let mut fused_tracks = LmbTrackSet::with_capacity(tracks_by_label.len());

    for (_label, sensor_tracks) in tracks_by_label {
        if sensor_tracks.len() == 1 {
            // Only one sensor has this track - use it directly
            let (_, track) = sensor_tracks[0];
            fused_tracks.push(track.clone());
        } else {
            // Multiple sensors - fuse
            let track_refs: Vec<&LmbTrack<T, N>> = sensor_tracks.iter().map(|(_, t)| *t).collect();
            let weights: Vec<T> = sensor_tracks
                .iter()
                .map(|(idx, _)| sensor_weights.get(*idx).copied().unwrap_or(T::one()))
                .collect();

            let fused = merger.merge(&track_refs, &weights);
            fused_tracks.push(fused);
        }
    }

    fused_tracks
}

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

// Type aliases are defined at usage site due to const generic limitations

// ============================================================================
// Builder for Multi-Sensor Filter
// ============================================================================

/// Builder for creating multi-sensor LMB filters.
#[cfg(feature = "alloc")]
pub struct MultisensorLmbFilterBuilder<T, Trans, Birth, const N: usize>
where
    T: RealField,
    Trans: TransitionModel<T, N>,
    Birth: LabeledBirthModel<T, N>,
{
    transition: Trans,
    birth: Birth,
    sensor_weights: Vec<T>,
    _marker: PhantomData<T>,
}

#[cfg(feature = "alloc")]
impl<T, Trans, Birth, const N: usize> MultisensorLmbFilterBuilder<T, Trans, Birth, N>
where
    T: RealField + Float + Copy,
    Trans: TransitionModel<T, N> + Clone,
    Birth: LabeledBirthModel<T, N> + Clone,
{
    /// Creates a new builder.
    pub fn new(transition: Trans, birth: Birth) -> Self {
        Self {
            transition,
            birth,
            sensor_weights: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Sets the sensor weights.
    pub fn with_sensor_weights(mut self, weights: Vec<T>) -> Self {
        self.sensor_weights = weights;
        self
    }

    /// Builds an AA-LMB (arithmetic average) filter.
    pub fn build_aa(
        self,
        max_components: usize,
    ) -> MultisensorLmbFilter<T, Trans, Birth, ArithmeticAverageMerger, N> {
        let num_sensors = self.sensor_weights.len().max(1);
        let weights = if self.sensor_weights.is_empty() {
            vec![T::one() / T::from(num_sensors).unwrap(); num_sensors]
        } else {
            self.sensor_weights
        };

        MultisensorLmbFilter {
            transition: self.transition,
            birth: self.birth,
            merger: ArithmeticAverageMerger::new(max_components),
            sensor_weights: weights,
            _marker: PhantomData,
        }
    }

    /// Builds a GA-LMB (geometric average) filter.
    pub fn build_ga(self) -> MultisensorLmbFilter<T, Trans, Birth, GeometricAverageMerger, N> {
        let num_sensors = self.sensor_weights.len().max(1);
        let weights = if self.sensor_weights.is_empty() {
            vec![T::one() / T::from(num_sensors).unwrap(); num_sensors]
        } else {
            self.sensor_weights
        };

        MultisensorLmbFilter {
            transition: self.transition,
            birth: self.birth,
            merger: GeometricAverageMerger::new(),
            sensor_weights: weights,
            _marker: PhantomData,
        }
    }

    /// Builds a PU-LMB (parallel update) filter.
    pub fn build_pu(
        self,
        prior_covariance: StateCovariance<T, N>,
    ) -> MultisensorLmbFilter<T, Trans, Birth, ParallelUpdateMerger<T, N>, N> {
        let num_sensors = self.sensor_weights.len().max(1);
        let weights = if self.sensor_weights.is_empty() {
            vec![T::one() / T::from(num_sensors).unwrap(); num_sensors]
        } else {
            self.sensor_weights
        };

        MultisensorLmbFilter {
            transition: self.transition,
            birth: self.birth,
            merger: ParallelUpdateMerger::new(prior_covariance),
            sensor_weights: weights,
            _marker: PhantomData,
        }
    }

    /// Builds an IC-LMB (iterated corrector) filter.
    pub fn build_ic(self) -> MultisensorLmbFilter<T, Trans, Birth, IteratedCorrectorMerger, N> {
        let num_sensors = self.sensor_weights.len().max(1);
        let weights = if self.sensor_weights.is_empty() {
            vec![T::one() / T::from(num_sensors).unwrap(); num_sensors]
        } else {
            self.sensor_weights
        };

        MultisensorLmbFilter {
            transition: self.transition,
            birth: self.birth,
            merger: IteratedCorrectorMerger::new(),
            sensor_weights: weights,
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::gaussian::GaussianState;
    use crate::types::labels::Label;
    use crate::types::spaces::StateVector;

    #[cfg(feature = "alloc")]
    #[test]
    fn test_fuse_sensor_tracks() {
        // Create tracks from two sensors
        let label = Label::new(0, 0);
        let mean1: StateVector<f64, 4> = StateVector::from_array([1.0, 0.0, 0.0, 0.0]);
        let mean2: StateVector<f64, 4> = StateVector::from_array([2.0, 0.0, 0.0, 0.0]);
        let cov: StateCovariance<f64, 4> = StateCovariance::identity();

        let track1 = LmbTrack::new(label, 0.8, GaussianState::new(1.0, mean1, cov));
        let track2 = LmbTrack::new(label, 0.6, GaussianState::new(1.0, mean2, cov));

        let state1 = LmbFilterState::<f64, 4, Updated>::from_tracks(vec![track1]);
        let state2 = LmbFilterState::<f64, 4, Updated>::from_tracks(vec![track2]);

        let merger = ArithmeticAverageMerger::new(10);
        let fused = fuse_sensor_tracks(&[state1, state2], &[0.5, 0.5], &merger);

        assert_eq!(fused.len(), 1);
        assert_eq!(fused.tracks[0].label, label);
        // Existence should be average
        assert!((fused.tracks[0].existence - 0.7).abs() < 1e-10);
    }
}
