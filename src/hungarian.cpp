#include "hungarian/hungarian.hpp"



double Hungarian::solve(const Matrix& costMatrix, Vector& assignment)
{
  // Initialize dimensions and matrices
  nRows_ = costMatrix.rows();
  nCols_ = costMatrix.cols();
  minDim_ = std::min(nRows_, nCols_);

  // Check for negative values
  if ((costMatrix.array() < 0).any()) {
    throw std::invalid_argument("All matrix elements must be non-negative");
  }

  // Initialize working matrices
  distMatrix_ = costMatrix;
  starMatrix_ = BoolMatrix::Zero(nRows_, nCols_);
  primeMatrix_ = BoolMatrix::Zero(nRows_, nCols_);
  coveredRows_ = BoolVector::Zero(nRows_);
  coveredColumns_ = BoolVector::Zero(nCols_);

  // Initialize the distance matrix
  initializeMatrix(distMatrix_);

  // Find initial assignment
  findInitialAssignment();

  // Main algorithm loop
  while (!findOptimalAssignment()) {
    updateMatrix();
  }

  // Build assignment vector
  assignment = Vector::Constant(nRows_, -1);
  for (int row = 0; row < nRows_; ++row) {
    for (int col = 0; col < nCols_; ++col) {
      if (starMatrix_(row, col)) {
        assignment(row) = col;
        break;
      }
    }
  }

  return computeCost(costMatrix, assignment);
}

void Hungarian::initializeMatrix(Matrix& distMatrix)
{
  if (nRows_ <= nCols_) {
    // Subtract row minimums
    for (int row = 0; row < nRows_; ++row) {
      double minVal = distMatrix.row(row).minCoeff();
      distMatrix.row(row).array() -= minVal;
    }
  } else {
    // Subtract column minimums
    for (int col = 0; col < nCols_; ++col) {
      double minVal = distMatrix.col(col).minCoeff();
      distMatrix.col(col).array() -= minVal;
    }
  }
}

void Hungarian::findInitialAssignment()
{
  const double epsilon = std::numeric_limits<double>::epsilon();

  coveredRows_.setZero();
  coveredColumns_.setZero();

  if (nRows_ <= nCols_) {
    // Find zeros row by row
    for (int row = 0; row < nRows_; ++row) {
      for (int col = 0; col < nCols_; ++col) {
        if (std::abs(distMatrix_(row, col)) < epsilon && !coveredColumns_(col)) {
          starMatrix_(row, col) = true;
          coveredColumns_(col) = true;
          break;
        }
      }
    }
  } else {
    // Find zeros column by column
    for (int col = 0; col < nCols_; ++col) {
      for (int row = 0; row < nRows_; ++row) {
        if (std::abs(distMatrix_(row, col)) < epsilon && !coveredRows_(row)) {
          starMatrix_(row, col) = true;
          coveredColumns_(col) = true;
          coveredRows_(row) = true;
          break;
        }
      }
    }
    coveredRows_.setZero(); // Reset for main algorithm
  }
}

bool Hungarian::findOptimalAssignment()
{
  // Cover columns containing starred zeros
  coveredColumns_.setZero();
  for (int col = 0; col < nCols_; ++col) {
    for (int row = 0; row < nRows_; ++row) {
      if (starMatrix_(row, col)) {
        coveredColumns_(col) = true;
        break;
      }
    }
  }

  // Check if we have optimal assignment
  int coveredCount = coveredColumns_.cast<int>().sum();
  if (coveredCount >= minDim_) {
    return true; // Algorithm finished
  }

  // Main loop: find uncovered zeros and process them
  while (true) {
    int zeroRow, zeroCol;
    if (!findUncoveredZero(zeroRow, zeroCol)) {
      break; // No more uncovered zeros, need to update matrix
    }

    // Prime the zero
    primeMatrix_(zeroRow, zeroCol) = true;

    // Check for starred zero in the same row
    int starCol;
    if (findStarInRow(zeroRow, starCol)) {
      // Cover this row and uncover the column of the starred zero
      coveredRows_(zeroRow) = true;
      coveredColumns_(starCol) = false;
    } else {
      // No starred zero in row, augment the path
      augmentPath(zeroRow, zeroCol);

      // Clear primes and covers
      primeMatrix_.setZero();
      coveredRows_.setZero();

      // Cover columns with starred zeros
      coveredColumns_.setZero();
      for (int col = 0; col < nCols_; ++col) {
        for (int row = 0; row < nRows_; ++row) {
          if (starMatrix_(row, col)) {
            coveredColumns_(col) = true;
            break;
          }
        }
      }

      // Check if optimal
      coveredCount = coveredColumns_.cast<int>().sum();
      if (coveredCount >= minDim_) {
        return true;
      }
    }
  }

  return false; // Need to update matrix
}

void Hungarian::augmentPath(int row, int col)
{
  // Create alternating path of starred and primed zeros
  std::vector<std::pair<int, int>> path;
  path.push_back({row, col});

  while (true) {
    // Find starred zero in current column
    int starRow;
    if (!findStarInColumn(col, starRow)) {
      break; // End of path
    }
    path.push_back({starRow, col});

    // Find primed zero in starred zero's row
    int primeCol;
    if (!findPrimeInRow(starRow, primeCol)) {
      break; // This shouldn't happen in a correct implementation
    }
    path.push_back({starRow, primeCol});
    col = primeCol;
  }

  // Augment the assignment along the path
  for (size_t i = 0; i < path.size(); ++i) {
    int r = path[i].first;
    int c = path[i].second;
    if (i % 2 == 0) {
      starMatrix_(r, c) = true;  // Star primed zeros
    } else {
      starMatrix_(r, c) = false; // Unstar starred zeros
    }
  }
}

void Hungarian::updateMatrix()
{
  // Find minimum uncovered value
  double h = findMinUncoveredValue();

  // Add h to covered rows
  for (int row = 0; row < nRows_; ++row) {
    if (coveredRows_(row)) {
      distMatrix_.row(row).array() += h;
    }
  }

  // Subtract h from uncovered columns
  for (int col = 0; col < nCols_; ++col) {
    if (!coveredColumns_(col)) {
      distMatrix_.col(col).array() -= h;
    }
  }
}

bool Hungarian::findUncoveredZero(int& row, int& col)
{
  const double epsilon = std::numeric_limits<double>::epsilon();

  for (int c = 0; c < nCols_; ++c) {
    if (!coveredColumns_(c)) {
      for (int r = 0; r < nRows_; ++r) {
        if (!coveredRows_(r) && std::abs(distMatrix_(r, c)) < epsilon) {
          row = r;
          col = c;
          return true;
        }
      }
    }
  }
  return false;
}

bool Hungarian::findStarInRow(int row, int& col)
{
  for (int c = 0; c < nCols_; ++c) {
    if (starMatrix_(row, c)) {
      col = c;
      return true;
    }
  }
  return false;
}

bool Hungarian::findStarInColumn(int col, int& row)
{
  for (int r = 0; r < nRows_; ++r) {
    if (starMatrix_(r, col)) {
      row = r;
      return true;
    }
  }
  return false;
}

bool Hungarian::findPrimeInRow(int row, int& col)
{
  for (int c = 0; c < nCols_; ++c) {
    if (primeMatrix_(row, c)) {
      col = c;
      return true;
    }
  }
  return false;
}

double Hungarian::findMinUncoveredValue()
{
  double minVal = std::numeric_limits<double>::max();
  bool found = false;

  for (int row = 0; row < nRows_; ++row) {
    if (!coveredRows_(row)) {
      for (int col = 0; col < nCols_; ++col) {
        if (!coveredColumns_(col)) {
          minVal = std::min(minVal, distMatrix_(row, col));
          found = true;
        }
      }
    }
  }

  return found ? minVal : 0.0;
}

double Hungarian::computeCost(const Matrix& originalMatrix, const Vector& assignment)
{
  double cost = 0.0;
  for (int row = 0; row < assignment.size(); ++row) {
    if (assignment(row) >= 0) {
      cost += originalMatrix(row, assignment(row));
    }
  }
  return cost;
}
