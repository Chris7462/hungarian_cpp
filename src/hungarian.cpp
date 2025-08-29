#include <cmath>
#include <limits>

#include "hungarian/hungarian.hpp"


//********************************************************//
// Main solve function - entry point for solving assignment problem
//********************************************************//
double Hungarian::solve(const Matrix & distMatrix, Vector & assignment)
{
  // Check for negative values - throw exception instead of returning -1
  if ((distMatrix.array() < 0).any()) {
    throw std::invalid_argument("All matrix elements have to be non-negative.");
  }

  // Get matrix dimensions
  nRows_ = distMatrix.rows();
  nCols_ = distMatrix.cols();

  // Initialize working matrices
  distMatrix_ = distMatrix;  // Working copy
  coveredColumns_ = BoolVector::Zero(nCols_);
  coveredRows_ = BoolVector::Zero(nRows_);
  starMatrix_ = BoolMatrix::Zero(nRows_, nCols_);
  primeMatrix_ = BoolMatrix::Zero(nRows_, nCols_);

  // Solve the assignment problem
  executeOptimalAssignment();

  // Build final assignment vector
  Vector finalAssignment = Vector::Constant(nRows_, -1);
  buildAssignmentVector(finalAssignment);

  // Copy result to output parameter
  assignment = finalAssignment;

  // Return total cost using original matrix
  return computeAssignmentCost(distMatrix, finalAssignment);
}

//********************************************************//
// Execute optimal solution for assignment problem using Munkres algorithm
//********************************************************//
void Hungarian::executeOptimalAssignment()
{
  // Preliminary steps - subtract minimum from rows or columns
  if (nRows_ <= nCols_) {
    // Subtract minimum from each row
    for (int row = 0; row < nRows_; ++row) {
      double minValue = distMatrix_.row(row).minCoeff();
      distMatrix_.row(row).array() -= minValue;
    }

    // Find zeros and star them
    for (int row = 0; row < nRows_; ++row) {
      for (int col = 0; col < nCols_; ++col) {
        if (std::abs(distMatrix_(row, col)) < std::numeric_limits<double>::epsilon()) {
          if (!coveredColumns_(col)) {
            starMatrix_(row, col) = true;
            coveredColumns_(col) = true;
            break;
          }
        }
      }
    }
  } else {
    // Subtract minimum from each column
    for (int col = 0; col < nCols_; ++col) {
      double minValue = distMatrix_.col(col).minCoeff();
      distMatrix_.col(col).array() -= minValue;
    }

    // Find zeros and star them
    for (int col = 0; col < nCols_; ++col) {
      for (int row = 0; row < nRows_; ++row) {
        if (std::abs(distMatrix_(row, col)) < std::numeric_limits<double>::epsilon()) {
          if (!coveredRows_(row)) {
            starMatrix_(row, col) = true;
            coveredColumns_(col) = true;
            coveredRows_(row) = true;
            break;
          }
        }
      }
    }
    // Reset covered rows for next steps
    coveredRows_.setZero();
  }

  checkOptimalityAndProceed();
}

//********************************************************//
// Build assignment vector from starred zeros
//********************************************************//
void Hungarian::buildAssignmentVector(Vector & assignment)
{
  for (int row = 0; row < nRows_; ++row) {
    for (int col = 0; col < nCols_; ++col) {
      if (starMatrix_(row, col)) {
        assignment(row) = col;
        break;
      }
    }
  }
}

//********************************************************//
// Compute total cost of assignment using original matrix
//********************************************************//
double Hungarian::computeAssignmentCost(const Matrix & originalMatrix, const Vector & assignment)
{
  double cost = 0.0;
  for (int row = 0; row < assignment.size(); ++row) {
    int col = assignment(row);
    if (col >= 0) {
      cost += originalMatrix(row, col);
    }
  }
  return cost;
}

//********************************************************//
// Cover every column containing a starred zero
//********************************************************//
void Hungarian::coverColumnsWithStars()
{
  // Cover every column containing a starred zero
  for (int col = 0; col < nCols_; ++col) {
    if (starMatrix_.col(col).any()) {
      coveredColumns_(col) = true;
    }
  }

  checkOptimalityAndProceed();
}

//********************************************************//
// Check if optimal assignment found, proceed accordingly
//********************************************************//
void Hungarian::checkOptimalityAndProceed()
{
  // Count covered columns and get minimum dimension
  int nCoveredColumns = coveredColumns_.cast<int>().sum();
  int minDim = std::min(nRows_, nCols_);

  if (nCoveredColumns == minDim) {
    // Algorithm finished - assignment is built in solve()
    return;
  } else {
    findAndProcessUncoveredZeros();
  }
}

//********************************************************//
// Find and process uncovered zeros
//********************************************************//
void Hungarian::findAndProcessUncoveredZeros()
{
  bool zerosFound = true;

  while (zerosFound) {
    zerosFound = false;

    for (int col = 0; col < nCols_ && !zerosFound; ++col) {
      if (!coveredColumns_(col)) {
        for (int row = 0; row < nRows_; ++row) {
          if (!coveredRows_(row) && std::abs(distMatrix_(row, col)) < std::numeric_limits<double>::epsilon()) {
            // Prime zero
            primeMatrix_(row, col) = true;

            // Find starred zero in current row
            int starCol = -1;
            for (int c = 0; c < nCols_; c++) {
              if (starMatrix_(row, c)) {
                starCol = c;
                break;
              }
            }

            if (starCol == -1) {
              // No starred zero found
              augmentAlternatingPath(row, col);
              return;
            } else {
              coveredRows_(row) = true;
              coveredColumns_(starCol) = false;
              zerosFound = true;
              break;
            }
          }
        }
      }
    }
  }

  updateMatrixValues();
}

//********************************************************//
// Augment assignment along alternating path
//********************************************************//
void Hungarian::augmentAlternatingPath(int row, int col)
{
  // Generate temporary copy of starMatrix
  BoolMatrix newStarMatrix = starMatrix_;

  // Star current zero
  newStarMatrix(row, col) = true;

  // Find starred zero in current column
  int starCol = col;
  int starRow = findStarInColumn(starCol);

  while (starRow >= 0) {
    // Unstar the starred zero
    newStarMatrix(starRow, starCol) = false;

    // Find primed zero in current row
    int primeCol = findPrimeInRow(starRow);

    if (primeCol >= 0) {
      // Star the primed zero
      newStarMatrix(starRow, primeCol) = true;

      // Find starred zero in current column
      starCol = primeCol;
      starRow = findStarInColumn(starCol);
    } else {
      break;
    }
  }

  // Use temporary copy as new starMatrix
  // Delete all primes, uncover all rows
  primeMatrix_.setZero();
  starMatrix_ = newStarMatrix;
  coveredRows_.setZero();

  coverColumnsWithStars();
}

//********************************************************//
// Update matrix values by adding/subtracting minimum uncovered value
//********************************************************//
void Hungarian::updateMatrixValues()
{
  // Find smallest uncovered element h
  double h = std::numeric_limits<double>::max();
  for (int row = 0; row < nRows_; ++row) {
    if (!coveredRows_(row)) {
      for (int col = 0; col < nCols_; ++col) {
        if (!coveredColumns_(col)) {
          if (distMatrix_(row, col) < h) {
            h = distMatrix_(row, col);
          }
        }
      }
    }
  }

  // Add h to each covered row
  for (int row = 0; row < nRows_; ++row) {
    if (coveredRows_(row)) {
      distMatrix_.row(row).array() += h;
    }
  }

  // Subtract h from each uncovered column
  for (int col = 0; col < nCols_; ++col) {
    if (!coveredColumns_(col)) {
      distMatrix_.col(col).array() -= h;
    }
  }

  findAndProcessUncoveredZeros();
}

//********************************************************//
// Find starred zero in given row, return column index or -1 if not found
//********************************************************//
int Hungarian::findStarInRow(int row) const
{
  for (int col = 0; col < nCols_; ++col) {
    if (starMatrix_(row, col)) {
      return col;
    }
  }
  return -1;
}

//********************************************************//
// Find starred zero in given column, return row index or -1 if not found
//********************************************************//
int Hungarian::findStarInColumn(int col) const
{
  for (int row = 0; row < nRows_; ++row) {
    if (starMatrix_(row, col)) {
      return row;
    }
  }
  return -1;
}

//********************************************************//
// Find primed zero in given row, return column index or -1 if not found
//********************************************************//
int Hungarian::findPrimeInRow(int row) const
{
  for (int col = 0; col < nCols_; ++col) {
    if (primeMatrix_(row, col)) {
      return col;
    }
  }
  return -1;
}
