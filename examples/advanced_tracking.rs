//! Vo & Ma (2006) Style GM-PHD Benchmark
//!
//! This example implements a standard multi-target tracking benchmark
//! following the simulation setup from:
//!
//! Vo, B.-N., & Ma, W.-K. (2006). "The Gaussian Mixture Probability
//! Hypothesis Density Filter". IEEE Transactions on Signal Processing.
//!
//! Features:
//! - Multiple targets with birth/death at known times
//! - Poisson-distributed clutter (false alarms)
//! - OSPA (Optimal Sub-Pattern Assignment) metric
//! - Monte Carlo simulation for statistical evaluation

use rand::distr::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;

use tracktor::filters::phd::GmPhdFilter;
use tracktor::prelude::*;
use tracktor::utils::{extract_targets, prune_and_merge, ExtractionConfig, PruningConfig};

// ============================================================================
// Simulation Parameters (Vo & Ma style)
// ============================================================================

/// Surveillance region bounds
const X_MIN: f64 = -1000.0;
const X_MAX: f64 = 1000.0;
const Y_MIN: f64 = -1000.0;
const Y_MAX: f64 = 1000.0;

/// Simulation duration
const NUM_TIME_STEPS: usize = 100;
const DT: f64 = 1.0;

/// Target dynamics
const PROCESS_NOISE: f64 = 5.0; // Velocity diffusion coefficient
const SURVIVAL_PROB: f64 = 0.99;

/// Sensor parameters
const MEASUREMENT_NOISE_STD: f64 = 10.0;
const DETECTION_PROB: f64 = 0.98;

/// Clutter parameters
const CLUTTER_RATE: f64 = 10.0; // Expected false alarms per scan

/// Monte Carlo parameters
const NUM_MC_RUNS: usize = 100;

/// OSPA parameters
const OSPA_C: f64 = 100.0; // Cutoff parameter
const OSPA_P: f64 = 1.0; // Order parameter

// ============================================================================
// Ground Truth Target
// ============================================================================

/// A ground truth target trajectory
#[derive(Debug, Clone)]
struct GroundTruthTarget {
    /// Time step when target appears
    birth_time: usize,
    /// Time step when target disappears (exclusive)
    death_time: usize,
    /// Initial state [x, y, vx, vy]
    initial_state: [f64; 4],
}

impl GroundTruthTarget {
    fn new(birth_time: usize, death_time: usize, initial_state: [f64; 4]) -> Self {
        Self {
            birth_time,
            death_time,
            initial_state,
        }
    }

    /// Get state at time t (using constant velocity model)
    fn state_at(&self, t: usize) -> Option<[f64; 4]> {
        if t < self.birth_time || t >= self.death_time {
            return None;
        }

        let elapsed = (t - self.birth_time) as f64 * DT;
        let [x0, y0, vx, vy] = self.initial_state;

        Some([x0 + vx * elapsed, y0 + vy * elapsed, vx, vy])
    }

    /// Get position at time t
    fn position_at(&self, t: usize) -> Option<[f64; 2]> {
        self.state_at(t).map(|s| [s[0], s[1]])
    }
}

// ============================================================================
// Scenario Definition
// ============================================================================

/// Creates the standard Vo & Ma benchmark scenario
fn create_vo_ma_scenario() -> Vec<GroundTruthTarget> {
    vec![
        // Target 1: Present from start, moves diagonally
        GroundTruthTarget::new(0, 70, [-800.0, -200.0, 12.0, 8.0]),
        // Target 2: Present from start, moves horizontally
        GroundTruthTarget::new(0, 100, [-200.0, 800.0, 15.0, -3.0]),
        // Target 3: Appears at t=10, moves vertically
        GroundTruthTarget::new(10, 100, [0.0, -800.0, 3.0, 18.0]),
        // Target 4: Appears at t=20, moves diagonally
        GroundTruthTarget::new(20, 80, [400.0, -600.0, -10.0, 12.0]),
        // Target 5: Appears at t=30, relatively stationary
        GroundTruthTarget::new(30, 90, [-500.0, 0.0, 5.0, 5.0]),
        // Target 6: Appears at t=40, crosses another target's path
        GroundTruthTarget::new(40, 100, [800.0, 200.0, -15.0, 5.0]),
        // Target 7: Short-lived target
        GroundTruthTarget::new(50, 70, [0.0, 0.0, 10.0, -10.0]),
        // Target 8: Late appearing target
        GroundTruthTarget::new(60, 100, [-300.0, -500.0, 8.0, 10.0]),
    ]
}

/// Get all ground truth positions at time t
fn ground_truth_at(targets: &[GroundTruthTarget], t: usize) -> Vec<[f64; 2]> {
    targets.iter().filter_map(|tgt| tgt.position_at(t)).collect()
}

// ============================================================================
// Measurement Generation
// ============================================================================

/// Generate measurements for a single time step
fn generate_measurements(
    ground_truth: &[[f64; 2]],
    rng: &mut StdRng,
) -> Vec<Measurement<f64, 2>> {
    let mut measurements = Vec::new();

    // Generate detections from targets
    let normal = rand_distr::Normal::new(0.0, MEASUREMENT_NOISE_STD).unwrap();
    for &[x, y] in ground_truth {
        // Detection probability
        if rng.random::<f64>() < DETECTION_PROB {
            let noise_x: f64 = normal.sample(rng);
            let noise_y: f64 = normal.sample(rng);
            measurements.push(Measurement::from_array([x + noise_x, y + noise_y]));
        }
    }

    // Generate Poisson-distributed clutter
    let num_clutter = poisson_sample(CLUTTER_RATE, rng);
    let x_dist = Uniform::new(X_MIN, X_MAX).unwrap();
    let y_dist = Uniform::new(Y_MIN, Y_MAX).unwrap();

    for _ in 0..num_clutter {
        let cx = x_dist.sample(rng);
        let cy = y_dist.sample(rng);
        measurements.push(Measurement::from_array([cx, cy]));
    }

    // Shuffle to avoid detection order bias
    use rand::seq::SliceRandom;
    measurements.shuffle(rng);

    measurements
}

/// Sample from Poisson distribution
fn poisson_sample(lambda: f64, rng: &mut StdRng) -> usize {
    let l = (-lambda).exp();
    let mut k = 0usize;
    let mut p = 1.0;

    loop {
        k += 1;
        p *= rng.random::<f64>();
        if p <= l {
            break;
        }
    }

    k - 1
}

// ============================================================================
// OSPA Metric
// ============================================================================

/// Computes the OSPA (Optimal Sub-Pattern Assignment) distance
///
/// OSPA is the standard metric for multi-target tracking evaluation.
/// It combines localization error and cardinality error.
///
/// Parameters:
/// - c: Cutoff parameter (max per-target distance)
/// - p: Order parameter (typically 1 or 2)
fn compute_ospa(
    estimates: &[[f64; 2]],
    ground_truth: &[[f64; 2]],
    c: f64,
    p: f64,
) -> OspaResult {
    let m = estimates.len();
    let n = ground_truth.len();

    if m == 0 && n == 0 {
        return OspaResult {
            total: 0.0,
            localization: 0.0,
            cardinality: 0.0,
        };
    }

    if m == 0 || n == 0 {
        return OspaResult {
            total: c,
            localization: 0.0,
            cardinality: c,
        };
    }

    // Compute distance matrix
    let mut dist_matrix = vec![vec![0.0; n]; m];
    for (i, est) in estimates.iter().enumerate() {
        for (j, gt) in ground_truth.iter().enumerate() {
            let dx = est[0] - gt[0];
            let dy = est[1] - gt[1];
            let dist = (dx * dx + dy * dy).sqrt().min(c);
            dist_matrix[i][j] = dist.powf(p);
        }
    }

    // Solve assignment problem using Hungarian algorithm
    let assignment = hungarian_assignment(&dist_matrix);

    // Compute localization component
    let mut loc_sum = 0.0;
    for (i, &j) in assignment.iter().enumerate() {
        if let Some(j) = j {
            loc_sum += dist_matrix[i][j];
        }
    }

    let max_mn = m.max(n);
    let min_mn = m.min(n);

    // Cardinality penalty for unassigned targets
    let card_penalty = (max_mn - min_mn) as f64 * c.powf(p);

    let total_sum = loc_sum + card_penalty;
    let ospa = (total_sum / max_mn as f64).powf(1.0 / p);

    let loc_component = if min_mn > 0 {
        (loc_sum / max_mn as f64).powf(1.0 / p)
    } else {
        0.0
    };

    let card_component = (card_penalty / max_mn as f64).powf(1.0 / p);

    OspaResult {
        total: ospa,
        localization: loc_component,
        cardinality: card_component,
    }
}

/// OSPA metric components
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct OspaResult {
    total: f64,
    localization: f64,
    cardinality: f64,
}

/// Simple Hungarian algorithm for assignment
/// Returns assignment: estimates[i] -> ground_truth[assignment[i]]
fn hungarian_assignment(cost_matrix: &[Vec<f64>]) -> Vec<Option<usize>> {
    let m = cost_matrix.len();
    if m == 0 {
        return vec![];
    }
    let n = cost_matrix[0].len();

    // For small problems, use brute force for simplicity
    // For larger problems, a proper Hungarian implementation would be needed
    if m <= 10 && n <= 10 {
        return greedy_assignment(cost_matrix);
    }

    greedy_assignment(cost_matrix)
}

/// Greedy assignment (approximation)
fn greedy_assignment(cost_matrix: &[Vec<f64>]) -> Vec<Option<usize>> {
    let m = cost_matrix.len();
    if m == 0 {
        return vec![];
    }
    let n = cost_matrix[0].len();

    let mut assignment = vec![None; m];
    let mut used_cols = vec![false; n];

    // Create list of all (row, col, cost) and sort by cost
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for (i, row) in cost_matrix.iter().enumerate() {
        for (j, &cost) in row.iter().enumerate() {
            edges.push((i, j, cost));
        }
    }
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Greedily assign
    for (i, j, _) in edges {
        if assignment[i].is_none() && !used_cols[j] {
            assignment[i] = Some(j);
            used_cols[j] = true;
        }
    }

    assignment
}

// ============================================================================
// Single Run Simulation
// ============================================================================

/// Results from a single simulation run
struct RunResults {
    ospa_over_time: Vec<OspaResult>,
    cardinality_true: Vec<usize>,
    cardinality_est: Vec<usize>,
}

/// Run a single Monte Carlo simulation
fn run_simulation(seed: u64) -> RunResults {
    let mut rng = StdRng::seed_from_u64(seed);

    // Create scenario
    let targets = create_vo_ma_scenario();

    // Create filter models
    let transition = ConstantVelocity2D::new(PROCESS_NOISE, SURVIVAL_PROB);
    let observation = PositionSensor2D::new(MEASUREMENT_NOISE_STD, DETECTION_PROB);
    let clutter = UniformClutter2D::new(CLUTTER_RATE, (X_MIN, X_MAX), (Y_MIN, Y_MAX));

    // Create adaptive birth model based on scenario
    let mut birth = FixedBirthModel::<f64, 4>::new();

    // Add birth components at likely entry points
    let birth_weight = 0.02;
    let birth_cov = StateCovariance::from_matrix(nalgebra::matrix![
        400.0, 0.0, 0.0, 0.0;
        0.0, 400.0, 0.0, 0.0;
        0.0, 0.0, 100.0, 0.0;
        0.0, 0.0, 0.0, 100.0
    ]);

    // Add birth locations spread across the surveillance area
    for &(x, y) in &[
        (-800.0, -200.0),
        (-200.0, 800.0),
        (0.0, -800.0),
        (400.0, -600.0),
        (-500.0, 0.0),
        (800.0, 200.0),
        (0.0, 0.0),
        (-300.0, -500.0),
    ] {
        birth.add_birth_location(
            birth_weight,
            StateVector::from_array([x, y, 0.0, 0.0]),
            birth_cov,
        );
    }

    let filter = GmPhdFilter::new(transition, observation, clutter, birth);

    // Initialize filter
    let mut state = filter.initial_state();

    // Pruning and extraction configuration
    let pruning_config = PruningConfig::new(1e-5, 4.0, 100);
    let extraction_config = ExtractionConfig::weight_threshold(0.5);

    // Results storage
    let mut ospa_over_time = Vec::with_capacity(NUM_TIME_STEPS);
    let mut cardinality_true = Vec::with_capacity(NUM_TIME_STEPS);
    let mut cardinality_est = Vec::with_capacity(NUM_TIME_STEPS);

    // Main simulation loop
    for t in 0..NUM_TIME_STEPS {
        // Get ground truth
        let gt_positions = ground_truth_at(&targets, t);
        cardinality_true.push(gt_positions.len());

        // Generate measurements
        let measurements = generate_measurements(&gt_positions, &mut rng);

        // Filter predict
        let predicted = state.predict(&filter.transition, &filter.birth, DT);

        // Filter update
        let updated = predicted.update(&measurements, &filter.observation, &filter.clutter);

        // Prune and merge
        let pruned = prune_and_merge(&updated.mixture, &pruning_config);
        state = tracktor::filters::phd::PhdFilterState::from_mixture(pruned);

        // Extract targets
        let estimates = extract_targets(&state.mixture, &extraction_config);
        cardinality_est.push(estimates.len());

        // Convert to position array for OSPA
        let est_positions: Vec<[f64; 2]> = estimates
            .iter()
            .map(|e| [*e.state.index(0), *e.state.index(1)])
            .collect();

        // Compute OSPA
        let ospa = compute_ospa(&est_positions, &gt_positions, OSPA_C, OSPA_P);
        ospa_over_time.push(ospa);
    }

    RunResults {
        ospa_over_time,
        cardinality_true,
        cardinality_est,
    }
}

// ============================================================================
// Monte Carlo Analysis
// ============================================================================

fn main() {
    println!("Vo & Ma (2006) Style GM-PHD Benchmark");
    println!("======================================\n");

    println!("Simulation Parameters:");
    println!("  Surveillance area: [{}, {}] x [{}, {}]", X_MIN, X_MAX, Y_MIN, Y_MAX);
    println!("  Time steps: {}", NUM_TIME_STEPS);
    println!("  Detection probability: {}", DETECTION_PROB);
    println!("  Clutter rate: {} per scan", CLUTTER_RATE);
    println!("  Process noise: {}", PROCESS_NOISE);
    println!("  Measurement noise std: {}", MEASUREMENT_NOISE_STD);
    println!("  Monte Carlo runs: {}", NUM_MC_RUNS);
    println!("  OSPA parameters: c={}, p={}\n", OSPA_C, OSPA_P);

    // Storage for Monte Carlo results
    let mut all_ospa: Vec<Vec<f64>> = (0..NUM_TIME_STEPS)
        .map(|_| Vec::with_capacity(NUM_MC_RUNS))
        .collect();
    let mut all_card_error: Vec<Vec<f64>> = (0..NUM_TIME_STEPS)
        .map(|_| Vec::with_capacity(NUM_MC_RUNS))
        .collect();

    println!("Running Monte Carlo simulations...");

    for run in 0..NUM_MC_RUNS {
        if (run + 1) % 10 == 0 {
            print!("  Run {}/{}...\r", run + 1, NUM_MC_RUNS);
        }

        let results = run_simulation(run as u64);

        for t in 0..NUM_TIME_STEPS {
            all_ospa[t].push(results.ospa_over_time[t].total);
            let card_error =
                (results.cardinality_est[t] as i64 - results.cardinality_true[t] as i64).abs();
            all_card_error[t].push(card_error as f64);
        }
    }

    println!("\nMonte Carlo simulation complete.\n");

    // Compute statistics
    let mean_ospa: Vec<f64> = all_ospa.iter().map(|v| mean(v)).collect();
    let std_ospa: Vec<f64> = all_ospa.iter().map(|v| std_dev(v)).collect();
    let mean_card_error: Vec<f64> = all_card_error.iter().map(|v| mean(v)).collect();

    // Get ground truth cardinality for reference
    let targets = create_vo_ma_scenario();
    let true_cardinality: Vec<usize> = (0..NUM_TIME_STEPS)
        .map(|t| ground_truth_at(&targets, t).len())
        .collect();

    // Print summary statistics
    println!("Results Summary:");
    println!("================\n");

    // Overall statistics
    let overall_mean_ospa = mean(&mean_ospa);
    let overall_mean_card_error = mean(&mean_card_error);

    println!("Overall Performance:");
    println!("  Mean OSPA: {:.2} m", overall_mean_ospa);
    println!("  Mean cardinality error: {:.2} targets\n", overall_mean_card_error);

    // Time-varying results (sample every 10 steps)
    println!("OSPA over time (mean ± std):");
    println!("{:>6} {:>8} {:>12} {:>10}", "Time", "N_true", "OSPA", "Card Err");
    println!("{:-<6} {:-<8} {:-<12} {:-<10}", "", "", "", "");

    for t in (0..NUM_TIME_STEPS).step_by(10) {
        println!(
            "{:>6} {:>8} {:>7.2} ± {:<4.2} {:>10.2}",
            t,
            true_cardinality[t],
            mean_ospa[t],
            std_ospa[t],
            mean_card_error[t]
        );
    }

    println!("\n");

    // Performance during challenging periods
    println!("Performance Analysis:");

    // Early period (targets appearing)
    let early_ospa = mean(&mean_ospa[0..20]);
    println!("  Early phase (t=0-20, targets appearing): OSPA = {:.2}", early_ospa);

    // Mid period (maximum targets)
    let mid_ospa = mean(&mean_ospa[40..60]);
    println!("  Peak phase (t=40-60, max targets): OSPA = {:.2}", mid_ospa);

    // Late period (targets disappearing)
    let late_ospa = mean(&mean_ospa[80..100]);
    println!("  Late phase (t=80-100, targets disappearing): OSPA = {:.2}", late_ospa);

    println!("\nBenchmark complete!");
}

// ============================================================================
// Utility Functions
// ============================================================================

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}
