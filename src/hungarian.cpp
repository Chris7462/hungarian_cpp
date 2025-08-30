#include <cmath>
#include <stdexcept>
#include <vector>

#include "hungarian/hungarian.hpp"


//********************************************************//
// Main solve function - clean entry point with min/max support
//********************************************************//
double Hungarian::solve(const MatrixXd & matrix, VectorXi & assignment, bool minimize)
{
  // Validate input matrix dimensions
  if (matrix.rows() <= 0 || matrix.cols() <= 0) {
    throw std::invalid_argument("Matrix dimensions must be positive.");
  }

  // Check for infinite or NaN values
  if (!matrix.allFinite()) {
    throw std::invalid_argument("Matrix contains infinite or NaN values.");
  }

  // For minimization, check for negative values (cost matrices should be non-negative)
  // For maximization, we allow negative profits
  if (minimize && (matrix.array() < 0).any()) {
    throw std::invalid_argument("Cost matrix elements must be non-negative for minimization.");
  }

  // Initialize algorithm state
  initializeAlgorithm(matrix, minimize);

  // Execute main algorithm loop
  executeMainLoop();

  // Extract results
  buildAssignmentVector(assignment);

  return computeTotalCost(matrix, assignment);
}

//********************************************************//
// Initialize all algorithm state and data structures
//********************************************************//
void Hungarian::initializeAlgorithm(const MatrixXd & matrix, bool minimize)
{
  // Cache dimensions
  nRows_ = matrix.rows();
  nCols_ = matrix.cols();

  // Transform matrix if maximizing profits
  if (minimize) {
    workingMatrix_ = matrix;  // Use matrix as-is for minimization
  } else {
    workingMatrix_ = transformProfitToCost(matrix);  // Transform for maximization
  }

  // Initialize state matrices
  starMatrix_ = BoolMatrix::Zero(nRows_, nCols_);
  primeMatrix_ = BoolMatrix::Zero(nRows_, nCols_);
  coveredRows_ = BoolVector::Zero(nRows_);
  coveredColumns_ = BoolVector::Zero(nCols_);
}

//********************************************************//
// Transform profit matrix to cost matrix for maximization
//********************************************************//
Hungarian::MatrixXd Hungarian::transformProfitToCost(const MatrixXd & profitMatrix) const
{
  // Find the maximum profit value
  double maxProfit = profitMatrix.maxCoeff();

  // Transform: cost[i][j] = max_profit - profit[i][j]
  // This ensures all costs are non-negative and preserves the optimal assignment
  return MatrixXd::Constant(profitMatrix.rows(), profitMatrix.cols(), maxProfit) - profitMatrix;
}

//********************************************************//
// Main algorithm loop - clean step-by-step execution
//********************************************************//
void Hungarian::executeMainLoop()
{
  step1_ReduceMatrix();
  step2_StarZeros();
  step3_CoverStarredColumns();

  // Add safety counter to prevent infinite loops
  int maxIterations = nRows_ * nCols_ * 10; // Conservative upper bound
  int iterations = 0;

  // Main loop - continue until optimal solution found
  while (!isOptimalSolutionFound()) {
    if (++iterations > maxIterations) {
      throw std::runtime_error("Algorithm failed to converge - possible implementation error");
    }
    step4_FindUncoveredZero();
  }
}

//********************************************************//
// Check if we have found the optimal solution
//********************************************************//
bool Hungarian::isOptimalSolutionFound() const
{
  int coveredCount = coveredColumns_.cast<int>().sum();
  int minDimension = std::min(nRows_, nCols_);
  return coveredCount >= minDimension;
}

//********************************************************//
// Step 1: Reduce matrix by subtracting row/column minimums
//********************************************************//
void Hungarian::step1_ReduceMatrix()
{
  if (nRows_ <= nCols_) {
    // Subtract minimum from each row
    for (int row = 0; row < nRows_; ++row) {
      double minValue = workingMatrix_.row(row).minCoeff();
      workingMatrix_.row(row).array() -= minValue;
    }
  } else {
    // Subtract minimum from each column
    for (int col = 0; col < nCols_; ++col) {
      double minValue = workingMatrix_.col(col).minCoeff();
      workingMatrix_.col(col).array() -= minValue;
    }
  }
}

//********************************************************//
// Step 2: Find zeros and star them (one per row/column)
//********************************************************//
void Hungarian::step2_StarZeros()
{
  if (nRows_ <= nCols_) {
    // Star zeros row by row
    for (int row = 0; row < nRows_; ++row) {
      for (int col = 0; col < nCols_; ++col) {
        if (std::abs(workingMatrix_(row, col)) < EPSILON && !coveredColumns_(col)) {
          starMatrix_(row, col) = true;
          coveredColumns_(col) = true;
          break;
        }
      }
    }
  } else {
    // Star zeros column by column
    for (int col = 0; col < nCols_; ++col) {
      for (int row = 0; row < nRows_; ++row) {
        if (std::abs(workingMatrix_(row, col)) < EPSILON && !coveredRows_(row)) {
          starMatrix_(row, col) = true;
          coveredColumns_(col) = true;
          coveredRows_(row) = true;
          break;
        }
      }
    }
    // Reset row coverage for next steps
    coveredRows_.setZero();
  }
}

//********************************************************//
// Step 3: Cover all columns containing starred zeros
//********************************************************//
void Hungarian::step3_CoverStarredColumns()
{
  coveredColumns_ = starMatrix_.colwise().any().transpose();
}

//********************************************************//
// Step 4: Find uncovered zero and decide next action
//********************************************************//
void Hungarian::step4_FindUncoveredZero()
{
  while (true) {
    auto [zeroRow, zeroCol] = findUncoveredZero();

    if (zeroRow == -1) {
      // No uncovered zero found - go to step 6
      step6_UpdateMatrix();
      return;
    }

    // Prime the uncovered zero
    primeMatrix_(zeroRow, zeroCol) = true;

    // Check if there's a starred zero in the same row
    int starCol = findStarInRow(zeroRow);

    if (starCol == -1) {
      // No starred zero in row - augment path (step 5)
      step5_AugmentPath(zeroRow, zeroCol);
      return;
    } else {
      // Cover this row and uncover the starred column
      coveredRows_(zeroRow) = true;
      coveredColumns_(starCol) = false;
    }
  }
}

//********************************************************//
// Step 5: Create alternating path and update stars
//********************************************************//
void Hungarian::step5_AugmentPath(const int startRow, const int startCol)
{
  // Build the alternating path first, then apply all changes
  std::vector<std::pair<int, int>> pathToStar;    // Positions to star
  std::vector<std::pair<int, int>> pathToUnstar;  // Positions to unstar

  // Start by starring the initial primed zero
  pathToStar.emplace_back(startRow, startCol);

  int currentCol = startCol;
  int starRow = findStarInColumn(currentCol);

  // Build alternating path: find star -> unstar it -> find prime -> star it
  while (starRow != -1) {
    // This starred zero will be unstarred
    pathToUnstar.emplace_back(starRow, currentCol);

    // Find the primed zero in the same row
    int primeCol = findPrimeInRow(starRow);

    if (primeCol != -1) {
      // This primed zero will be starred
      pathToStar.emplace_back(starRow, primeCol);

      // Continue the path from this column
      currentCol = primeCol;
      starRow = findStarInColumn(currentCol);
    } else {
      // No primed zero in this row - path ends
      break;
    }
  }

  // Apply all changes to the star matrix
  for (const auto & pos : pathToUnstar) {
    starMatrix_(pos.first, pos.second) = false;
  }
  for (const auto & pos : pathToStar) {
    starMatrix_(pos.first, pos.second) = true;
  }

  // Clear all primes and row coverage
  primeMatrix_.setZero();
  coveredRows_.setZero();

  // Go back to step 3
  step3_CoverStarredColumns();
}

//********************************************************//
// Step 6: Update matrix by adding/subtracting minimum uncovered value
//********************************************************//
void Hungarian::step6_UpdateMatrix()
{
  double minValue = findMinimumUncoveredValue();

  // Add minimum to covered rows
  for (int row = 0; row < nRows_; ++row) {
    if (coveredRows_(row)) {
      workingMatrix_.row(row).array() += minValue;
    }
  }

  // Subtract minimum from uncovered columns
  for (int col = 0; col < nCols_; ++col) {
    if (!coveredColumns_(col)) {
      workingMatrix_.col(col).array() -= minValue;
    }
  }
}

//********************************************************//
// Build final assignment vector from starred zeros
//********************************************************//
void Hungarian::buildAssignmentVector(VectorXi & assignment) const
{
  // Ensure assignment vector is properly sized
  if (assignment.size() != nRows_) {
    assignment = VectorXi::Constant(nRows_, -1);
  }

  for (int row = 0; row < nRows_; ++row) {
    assignment(row) = -1;  // Initialize to unassigned
    for (int col = 0; col < nCols_; ++col) {
      if (starMatrix_(row, col)) {
        assignment(row) = col;
        break;  // Only one star per row
      }
    }
  }
}

//********************************************************//
// Compute total assignment cost/profit using original matrix
//********************************************************//
double Hungarian::computeTotalCost(
  const MatrixXd & originalMatrix,
  const VectorXi & assignment) const
{
  // Validate assignment vector size
  if (assignment.size() != originalMatrix.rows()) {
    throw std::invalid_argument("Assignment vector size must match matrix row count.");
  }

  double totalCost = 0.0;
  for (int row = 0; row < assignment.size(); ++row) {
    const int col = assignment(row);
    if (col >= 0 && col < originalMatrix.cols()) {
      totalCost += originalMatrix(row, col);
    }
    // Note: col == -1 means this row is unassigned (valid for non-square matrices)
  }
  return totalCost;
}

//********************************************************//
// Utility methods for searching matrix elements
//********************************************************//
int Hungarian::findStarInRow(const int row) const
{
  // Validate row index
  if (row < 0 || row >= nRows_) {
    return -1;
  }

  for (int col = 0; col < nCols_; ++col) {
    if (starMatrix_(row, col)) {
      return col;
    }
  }
  return -1;
}

int Hungarian::findStarInColumn(const int col) const
{
  // Validate column index
  if (col < 0 || col >= nCols_) {
    return -1;
  }

  for (int row = 0; row < nRows_; ++row) {
    if (starMatrix_(row, col)) {
      return row;
    }
  }
  return -1;
}

int Hungarian::findPrimeInRow(const int row) const
{
  // Validate row index
  if (row < 0 || row >= nRows_) {
    return -1;
  }

  for (int col = 0; col < nCols_; ++col) {
    if (primeMatrix_(row, col)) {
      return col;
    }
  }
  return -1;
}

std::pair<int, int> Hungarian::findUncoveredZero() const
{
  for (int row = 0; row < nRows_; ++row) {
    if (!coveredRows_(row)) {
      for (int col = 0; col < nCols_; ++col) {
        if (!coveredColumns_(col) && std::abs(workingMatrix_(row, col)) < EPSILON) {
          return {row, col};
        }
      }
    }
  }
  return {-1, -1};
}

double Hungarian::findMinimumUncoveredValue() const
{
  double minValue = std::numeric_limits<double>::max();
  bool foundUncoveredElement = false;

  for (int row = 0; row < nRows_; ++row) {
    if (!coveredRows_(row)) {
      for (int col = 0; col < nCols_; ++col) {
        if (!coveredColumns_(col)) {
          minValue = std::min(minValue, workingMatrix_(row, col));
          foundUncoveredElement = true;
        }
      }
    }
  }

  // Add this check
  if (!foundUncoveredElement) {
    throw std::runtime_error("Algorithm error: No uncovered elements found when updating matrix");
  }

  return minValue;
}
