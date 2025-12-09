//! Hungarian Algorithm for Optimal Assignment
//!
//! Implementation of the Hungarian (Kuhn-Munkres) algorithm for solving
//! the linear assignment problem in O(n³) time.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::TracktorError;

/// Result of an assignment problem.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    /// Assignment mapping: row i is assigned to column assignment[i]
    /// None means the row is unassigned
    #[cfg(feature = "alloc")]
    pub mapping: Vec<Option<usize>>,
    /// Total cost of the assignment
    pub cost: f64,
}

#[cfg(feature = "alloc")]
impl Assignment {
    /// Creates a new assignment with the given mapping and cost.
    pub fn new(mapping: Vec<Option<usize>>, cost: f64) -> Self {
        Self { mapping, cost }
    }

    /// Returns the number of assigned pairs.
    pub fn num_assigned(&self) -> usize {
        self.mapping.iter().filter(|x| x.is_some()).count()
    }

    /// Returns an iterator over (row, col) pairs.
    pub fn pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.mapping
            .iter()
            .enumerate()
            .filter_map(|(row, col)| col.map(|c| (row, c)))
    }
}

/// Cost matrix for assignment problems.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct CostMatrix {
    /// Row-major cost data
    data: Vec<f64>,
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
}

#[cfg(feature = "alloc")]
impl CostMatrix {
    /// Creates a cost matrix from row-major data.
    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Result<Self, TracktorError> {
        if data.len() != rows * cols {
            return Err(TracktorError::AssignmentFailed);
        }
        Ok(Self { data, rows, cols })
    }

    /// Creates a cost matrix filled with a value.
    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Self {
            data: vec![value; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates a zero-filled cost matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::filled(rows, cols, 0.0)
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Gets the cost at (row, col).
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    /// Sets the cost at (row, col).
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    /// Returns a mutable reference to the cost at (row, col).
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        &mut self.data[row * self.cols + col]
    }
}

/// Solves the linear assignment problem using the Hungarian algorithm.
///
/// Given a cost matrix C, finds an assignment that minimizes the total cost.
/// This is a correct implementation of the Kuhn-Munkres algorithm.
///
/// # Arguments
/// - `cost`: The cost matrix (rows = workers, columns = jobs)
///
/// # Returns
/// An `Assignment` with the optimal row-to-column mapping.
#[cfg(feature = "alloc")]
pub fn hungarian(cost: &CostMatrix) -> Result<Assignment, TracktorError> {
    let n_rows = cost.rows();
    let n_cols = cost.cols();

    if n_rows == 0 || n_cols == 0 {
        return Ok(Assignment::new(vec![], 0.0));
    }

    // Make the matrix square by padding with large values
    let n = n_rows.max(n_cols);
    let large = 1e20_f64;

    let mut matrix = vec![large; n * n];
    for i in 0..n_rows {
        for j in 0..n_cols {
            matrix[i * n + j] = cost.get(i, j);
        }
    }

    // Potential vectors for rows and columns (dual variables)
    let mut u = vec![0.0_f64; n]; // Row potentials
    let mut v = vec![0.0_f64; n]; // Column potentials

    // Assignment arrays
    let mut row_assignment: Vec<Option<usize>> = vec![None; n];
    let mut col_assignment: Vec<Option<usize>> = vec![None; n];

    // Process each row
    for i in 0..n {
        // Start augmenting path from row i
        let mut min_to = vec![f64::INFINITY; n]; // Minimum reduced cost to reach each column
        let mut way = vec![None::<usize>; n]; // Previous column in augmenting path
        let mut used = vec![false; n]; // Columns visited in this iteration

        let mut cur_row = i;
        let mut cur_col: Option<usize> = None;

        // Find augmenting path using Dijkstra-like approach
        loop {
            let mut min_val = f64::INFINITY;
            let mut min_col = 0;

            // Find minimum reduced cost among unvisited columns
            for j in 0..n {
                if used[j] {
                    continue;
                }

                let reduced_cost = matrix[cur_row * n + j] - u[cur_row] - v[j];

                if reduced_cost < min_to[j] {
                    min_to[j] = reduced_cost;
                    way[j] = cur_col;
                }

                if min_to[j] < min_val {
                    min_val = min_to[j];
                    min_col = j;
                }
            }

            // Update potentials
            for j in 0..n {
                if used[j] {
                    u[col_assignment[j].unwrap()] += min_val;
                    v[j] -= min_val;
                } else {
                    min_to[j] -= min_val;
                }
            }
            u[i] += min_val;

            // Move to next column
            used[min_col] = true;
            cur_col = Some(min_col);

            // Check if this column is unassigned
            if col_assignment[min_col].is_none() {
                break;
            }

            // Follow assignment to next row
            cur_row = col_assignment[min_col].unwrap();
        }

        // Trace back augmenting path and update assignments
        while let Some(col) = cur_col {
            let prev_col = way[col];

            if let Some(pc) = prev_col {
                // Get the row that was assigned to prev_col
                col_assignment[col] = col_assignment[pc];
            } else {
                // This is the start of the path
                col_assignment[col] = Some(i);
            }

            cur_col = prev_col;
        }
    }

    // Build row_assignment from col_assignment
    for (j, col_asgn) in col_assignment.iter().enumerate().take(n) {
        if let Some(i) = col_asgn {
            row_assignment[*i] = Some(j);
        }
    }

    // Calculate total cost and build result
    let mut total_cost = 0.0;
    let mut result_mapping = Vec::with_capacity(n_rows);

    for (i, row_asgn) in row_assignment.iter().enumerate().take(n_rows) {
        if let Some(j) = row_asgn {
            if *j < n_cols {
                total_cost += cost.get(i, *j);
                result_mapping.push(Some(*j));
            } else {
                result_mapping.push(None);
            }
        } else {
            result_mapping.push(None);
        }
    }

    Ok(Assignment::new(result_mapping, total_cost))
}

/// Solves the assignment problem with a gating threshold.
///
/// Assignments with cost above the threshold are not made.
#[cfg(feature = "alloc")]
pub fn hungarian_gated(
    cost: &CostMatrix,
    gate_threshold: f64,
) -> Result<Assignment, TracktorError> {
    // Create a gated cost matrix
    let mut gated = CostMatrix::zeros(cost.rows(), cost.cols());
    let large = 1e20_f64;

    for i in 0..cost.rows() {
        for j in 0..cost.cols() {
            let c = cost.get(i, j);
            if c <= gate_threshold {
                gated.set(i, j, c);
            } else {
                gated.set(i, j, large);
            }
        }
    }

    let mut result = hungarian(&gated)?;

    // Filter out assignments above threshold
    for i in 0..result.mapping.len() {
        if let Some(j) = result.mapping[i] {
            if cost.get(i, j) > gate_threshold {
                result.mapping[i] = None;
            }
        }
    }

    // Recalculate cost
    result.cost = result.pairs().map(|(i, j)| cost.get(i, j)).sum();

    Ok(result)
}

/// Auction algorithm for assignment (alternative to Hungarian).
///
/// Better for sparse problems or when approximate solutions are acceptable.
#[cfg(feature = "alloc")]
pub fn auction(
    cost: &CostMatrix,
    epsilon: f64,
    max_iter: usize,
) -> Result<Assignment, TracktorError> {
    let n_rows = cost.rows();
    let n_cols = cost.cols();

    if n_rows == 0 || n_cols == 0 {
        return Ok(Assignment::new(vec![], 0.0));
    }

    // Initialize prices and assignments
    let mut prices = vec![0.0; n_cols];
    let mut row_assignment: Vec<Option<usize>> = vec![None; n_rows];
    let mut col_assignment: Vec<Option<usize>> = vec![None; n_cols];

    for _ in 0..max_iter {
        let mut any_unassigned = false;

        for i in 0..n_rows {
            if row_assignment[i].is_some() {
                continue;
            }
            any_unassigned = true;

            // Find best and second-best columns for row i
            let mut best_j = 0;
            let mut best_value = f64::NEG_INFINITY;
            let mut second_best = f64::NEG_INFINITY;

            for (j, price) in prices.iter().enumerate().take(n_cols) {
                let value = -cost.get(i, j) - price; // Negative because we minimize
                if value > best_value {
                    second_best = best_value;
                    best_value = value;
                    best_j = j;
                } else if value > second_best {
                    second_best = value;
                }
            }

            // Bid
            let bid_increment = best_value - second_best + epsilon;
            prices[best_j] += bid_increment;

            // Update assignments
            if let Some(prev_i) = col_assignment[best_j] {
                row_assignment[prev_i] = None;
            }
            row_assignment[i] = Some(best_j);
            col_assignment[best_j] = Some(i);
        }

        if !any_unassigned {
            break;
        }
    }

    // Calculate total cost
    let total_cost: f64 = row_assignment
        .iter()
        .enumerate()
        .filter_map(|(i, j)| j.map(|j| cost.get(i, j)))
        .sum();

    Ok(Assignment::new(row_assignment, total_cost))
}

// ============================================================================
// Murty's Algorithm for K-Best Assignments
// ============================================================================

/// A node in Murty's partition tree.
///
/// Each node represents a constrained assignment problem where some
/// assignments are required (fixed) and others are forbidden (excluded).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
struct MurtyNode {
    /// The assignment solution for this node
    assignment: Assignment,
    /// Row indices that have fixed assignments (row -> col)
    fixed: Vec<(usize, usize)>,
    /// Excluded assignments (row, col) that cannot be made
    excluded: Vec<(usize, usize)>,
    /// The row index up to which we've partitioned
    partition_row: usize,
}

#[cfg(feature = "alloc")]
impl MurtyNode {
    fn new(assignment: Assignment) -> Self {
        Self {
            assignment,
            fixed: Vec::new(),
            excluded: Vec::new(),
            partition_row: 0,
        }
    }

    fn with_constraints(
        assignment: Assignment,
        fixed: Vec<(usize, usize)>,
        excluded: Vec<(usize, usize)>,
        partition_row: usize,
    ) -> Self {
        Self {
            assignment,
            fixed,
            excluded,
            partition_row,
        }
    }
}

/// Solves a constrained assignment problem.
///
/// Some assignments are required (fixed) and some are forbidden (excluded).
/// Returns None if no valid assignment exists.
#[cfg(feature = "alloc")]
fn hungarian_constrained(
    cost: &CostMatrix,
    fixed: &[(usize, usize)],
    excluded: &[(usize, usize)],
) -> Option<Assignment> {
    let n_rows = cost.rows();
    let n_cols = cost.cols();

    if n_rows == 0 || n_cols == 0 {
        return Some(Assignment::new(vec![], 0.0));
    }

    // Check for conflicting constraints
    for &(r1, c1) in fixed {
        for &(r2, c2) in fixed {
            if r1 != r2 && c1 == c2 {
                // Two different rows fixed to same column - infeasible
                return None;
            }
        }
        // Check if fixed assignment is also excluded
        if excluded.contains(&(r1, c1)) {
            return None;
        }
    }

    // Create a modified cost matrix
    let large = 1e20_f64;
    let mut modified = CostMatrix::zeros(n_rows, n_cols);

    for i in 0..n_rows {
        for j in 0..n_cols {
            modified.set(i, j, cost.get(i, j));
        }
    }

    // Set excluded assignments to large cost
    for &(row, col) in excluded {
        if row < n_rows && col < n_cols {
            modified.set(row, col, large);
        }
    }

    // For fixed assignments, we need to ensure they're selected
    // We do this by setting all other costs in that row/column to large
    for &(row, col) in fixed {
        if row < n_rows && col < n_cols {
            // Set all other columns in this row to large cost
            for j in 0..n_cols {
                if j != col {
                    modified.set(row, j, large);
                }
            }
            // Set all other rows in this column to large cost
            for i in 0..n_rows {
                if i != row {
                    modified.set(i, col, large);
                }
            }
        }
    }

    // Solve using standard Hungarian
    let result = hungarian(&modified).ok()?;

    // Verify all fixed assignments were made
    for &(row, col) in fixed {
        if row < n_rows && result.mapping.get(row).copied().flatten() != Some(col) {
            return None; // Fixed assignment not satisfied
        }
    }

    // Verify no excluded assignments were made
    for &(row, col) in excluded {
        if row < n_rows && result.mapping.get(row).copied().flatten() == Some(col) {
            return None; // Excluded assignment was made
        }
    }

    // Recalculate cost using original cost matrix
    let real_cost: f64 = result
        .mapping
        .iter()
        .enumerate()
        .filter_map(|(i, &j)| j.filter(|&jj| jj < n_cols).map(|jj| cost.get(i, jj)))
        .sum();

    Some(Assignment::new(result.mapping, real_cost))
}

/// Finds the k-best assignments using Murty's algorithm.
///
/// Murty's algorithm partitions the solution space to efficiently find
/// the k best solutions without exhaustive enumeration.
///
/// # Arguments
/// - `cost`: The cost matrix
/// - `k`: The number of best assignments to find
///
/// # Returns
/// A vector of up to k assignments, sorted by cost (ascending).
#[cfg(feature = "alloc")]
pub fn murty_k_best(cost: &CostMatrix, k: usize) -> Vec<Assignment> {
    use alloc::collections::BinaryHeap;
    use core::cmp::Ordering;

    if k == 0 {
        return Vec::new();
    }

    // Wrapper for BinaryHeap (max-heap, but we want min-heap for costs)
    #[derive(Debug)]
    struct HeapNode(MurtyNode);

    impl PartialEq for HeapNode {
        fn eq(&self, other: &Self) -> bool {
            self.0.assignment.cost == other.0.assignment.cost
        }
    }

    impl Eq for HeapNode {}

    impl PartialOrd for HeapNode {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for HeapNode {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse ordering for min-heap behavior
            other
                .0
                .assignment
                .cost
                .partial_cmp(&self.0.assignment.cost)
                .unwrap_or(Ordering::Equal)
        }
    }

    let mut results: Vec<Assignment> = Vec::with_capacity(k);
    let mut heap: BinaryHeap<HeapNode> = BinaryHeap::new();

    // Get the optimal assignment
    let best = match hungarian(cost) {
        Ok(a) => a,
        Err(_) => return Vec::new(),
    };

    // Add root node to heap
    heap.push(HeapNode(MurtyNode::new(best)));

    while results.len() < k {
        // Get the best unexpanded node
        let node = match heap.pop() {
            Some(HeapNode(n)) => n,
            None => break, // No more solutions
        };

        // Add this solution to results
        results.push(node.assignment.clone());

        if results.len() >= k {
            break;
        }

        // Generate child nodes by partitioning
        let n_rows = cost.rows();
        let mapping = &node.assignment.mapping;

        // For each assigned pair starting from partition_row, create a child
        // where that assignment is excluded
        for row in node.partition_row..n_rows {
            if let Some(col) = mapping[row] {
                // Create child node where (row, col) is excluded
                let mut child_fixed = node.fixed.clone();
                let mut child_excluded = node.excluded.clone();

                // Fix all assignments before this row (same as in current solution)
                for (prev_row, prev_col) in mapping
                    .iter()
                    .enumerate()
                    .take(row)
                    .skip(node.partition_row)
                    .filter_map(|(r, c)| c.map(|col| (r, col)))
                {
                    if !child_fixed.iter().any(|&(r, _)| r == prev_row) {
                        child_fixed.push((prev_row, prev_col));
                    }
                }

                // Exclude the current (row, col) assignment
                child_excluded.push((row, col));

                // Solve the constrained problem
                if let Some(child_assignment) =
                    hungarian_constrained(cost, &child_fixed, &child_excluded)
                {
                    heap.push(HeapNode(MurtyNode::with_constraints(
                        child_assignment,
                        child_fixed,
                        child_excluded,
                        row,
                    )));
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn test_hungarian_simple() {
        // Simple 3x3 cost matrix
        let cost =
            CostMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();

        let result = hungarian(&cost).unwrap();

        // Optimal assignment: 0->2, 1->1, 2->0 (cost = 3+5+7 = 15)
        assert_eq!(result.num_assigned(), 3);
        assert!(
            (result.cost - 15.0).abs() < 0.01,
            "Expected cost 15.0, got {}",
            result.cost
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_hungarian_asymmetric() {
        // Cost matrix where diagonal is not optimal
        let cost =
            CostMatrix::from_vec(vec![10.0, 5.0, 13.0, 3.0, 15.0, 8.0, 7.0, 4.0, 12.0], 3, 3)
                .unwrap();

        let result = hungarian(&cost).unwrap();

        // Optimal: 0->1 (5), 1->0 (3), 2->2 (12) = 20
        // or: 0->2 (13), 1->0 (3), 2->1 (4) = 20
        assert_eq!(result.num_assigned(), 3);
        assert!(
            (result.cost - 20.0).abs() < 0.01,
            "Expected cost 20.0, got {}",
            result.cost
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_hungarian_rectangular() {
        // More rows than columns
        let cost = CostMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();

        let result = hungarian(&cost).unwrap();

        // Only 2 assignments possible
        assert!(result.num_assigned() <= 2);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_hungarian_gated() {
        let cost = CostMatrix::from_vec(vec![1.0, 100.0, 100.0, 2.0], 2, 2).unwrap();

        let result = hungarian_gated(&cost, 10.0).unwrap();

        // Both assignments should be made (1.0 and 2.0 are below threshold)
        assert_eq!(result.num_assigned(), 2);
        assert!((result.cost - 3.0).abs() < 0.1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_auction() {
        let cost =
            CostMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();

        let result = auction(&cost, 0.1, 100).unwrap();

        assert_eq!(result.num_assigned(), 3);
        // Auction is epsilon-optimal, should be within 3*epsilon of optimal (15.0)
        assert!(
            result.cost >= 14.5 && result.cost <= 15.5,
            "Auction cost {} not within bounds",
            result.cost
        );
    }

    // ============================================================================
    // Murty's Algorithm Tests
    // ============================================================================

    #[cfg(feature = "alloc")]
    #[test]
    fn test_murty_k1() {
        // With k=1, should return same result as Hungarian
        let cost =
            CostMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();

        let hungarian_result = hungarian(&cost).unwrap();
        let murty_results = murty_k_best(&cost, 1);

        assert_eq!(murty_results.len(), 1);
        assert!(
            (murty_results[0].cost - hungarian_result.cost).abs() < 0.01,
            "Murty k=1 should match Hungarian: {} vs {}",
            murty_results[0].cost,
            hungarian_result.cost
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_murty_k3_ordering() {
        // Test that results are returned in cost-ascending order
        let cost =
            CostMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();

        let results = murty_k_best(&cost, 3);

        assert!(results.len() >= 2, "Should find at least 2 solutions");

        // Verify costs are non-decreasing
        for i in 1..results.len() {
            assert!(
                results[i].cost >= results[i - 1].cost - 0.001,
                "Results should be ordered: cost[{}]={} >= cost[{}]={}",
                i,
                results[i].cost,
                i - 1,
                results[i - 1].cost
            );
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_murty_unique_solutions() {
        // Test that all returned solutions are unique
        let cost =
            CostMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();

        let results = murty_k_best(&cost, 5);

        // Check for duplicates
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                assert_ne!(
                    results[i].mapping, results[j].mapping,
                    "Solutions {} and {} should be different",
                    i, j
                );
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_murty_2x2() {
        // Simple 2x2 case where we can verify all solutions
        // Cost matrix:
        // [1, 10]
        // [10, 2]
        let cost = CostMatrix::from_vec(vec![1.0, 10.0, 10.0, 2.0], 2, 2).unwrap();

        let results = murty_k_best(&cost, 2);

        assert_eq!(results.len(), 2);

        // Best: (0,0) + (1,1) = 1 + 2 = 3
        assert!(
            (results[0].cost - 3.0).abs() < 0.01,
            "Best should be 3.0, got {}",
            results[0].cost
        );

        // Second best: (0,1) + (1,0) = 10 + 10 = 20
        assert!(
            (results[1].cost - 20.0).abs() < 0.01,
            "Second best should be 20.0, got {}",
            results[1].cost
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_murty_3x3_comprehensive() {
        // 3x3 case with known solutions
        // Cost matrix:
        // [1, 2, 3]
        // [4, 5, 6]
        // [7, 8, 9]
        let cost =
            CostMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();

        let results = murty_k_best(&cost, 6);

        // All 6 possible assignments for 3x3:
        // Optimal: 0->2, 1->1, 2->0 = 3+5+7 = 15
        assert!(
            (results[0].cost - 15.0).abs() < 0.01,
            "Optimal should be 15.0, got {}",
            results[0].cost
        );

        // Verify all solutions are valid assignments
        for (idx, result) in results.iter().enumerate() {
            let mut cols_used: Vec<bool> = vec![false; 3];
            for &col in &result.mapping {
                if let Some(c) = col {
                    assert!(
                        !cols_used[c],
                        "Solution {} has duplicate column assignment",
                        idx
                    );
                    cols_used[c] = true;
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_murty_rectangular() {
        // Rectangular matrix (more rows than columns)
        let cost = CostMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();

        let results = murty_k_best(&cost, 3);

        assert!(!results.is_empty(), "Should find at least one solution");

        // Each solution should assign at most 2 rows (since we have 2 columns)
        for result in &results {
            let assigned = result.mapping.iter().filter(|x| x.is_some()).count();
            assert!(
                assigned <= 2,
                "Should assign at most 2 rows, got {}",
                assigned
            );
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_murty_empty() {
        let cost = CostMatrix::zeros(0, 0);
        let results = murty_k_best(&cost, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cost, 0.0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_murty_k0() {
        let cost = CostMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let results = murty_k_best(&cost, 0);
        assert!(results.is_empty());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_hungarian_constrained_fixed() {
        // Test that fixed constraints work
        let cost = CostMatrix::from_vec(vec![1.0, 10.0, 10.0, 2.0], 2, 2).unwrap();

        // Without constraints: optimal is (0,0)+(1,1) = 3
        // Force (0,1) to be selected
        let fixed = vec![(0, 1)];
        let excluded = vec![];

        let result = hungarian_constrained(&cost, &fixed, &excluded);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.mapping[0], Some(1), "Row 0 should be fixed to col 1");
        // Cost should be 10 + 10 = 20 (forced suboptimal)
        assert!(
            (result.cost - 20.0).abs() < 0.01,
            "Cost should be 20.0, got {}",
            result.cost
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_hungarian_constrained_excluded() {
        // Test that excluded constraints work
        let cost = CostMatrix::from_vec(vec![1.0, 10.0, 10.0, 2.0], 2, 2).unwrap();

        // Without constraints: optimal is (0,0)+(1,1) = 3
        // Exclude (0,0) - forces second best
        let fixed = vec![];
        let excluded = vec![(0, 0)];

        let result = hungarian_constrained(&cost, &fixed, &excluded);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_ne!(
            result.mapping[0],
            Some(0),
            "Row 0 should not be assigned to col 0"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_hungarian_constrained_infeasible() {
        // Test that infeasible constraints return None
        let cost = CostMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();

        // Fix both rows to same column - infeasible
        let fixed = vec![(0, 0), (1, 0)];
        let excluded = vec![];

        let result = hungarian_constrained(&cost, &fixed, &excluded);

        assert!(result.is_none(), "Should be infeasible");
    }
}
