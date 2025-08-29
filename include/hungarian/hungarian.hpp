///////////////////////////////////////////////////////////////////////////////
// hungarian.hpp: Modern Hungarian Algorithm implementation using Eigen
// Based on Munkres algorithm for solving assignment problems
// Modernized version with Eigen library for better performance and readability
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>

class Hungarian
{
public:
  // Helper type definitions for better readability
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXi;
  using BoolMatrix = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
  using BoolVector = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

  Hungarian() = default;
  ~Hungarian() = default;

  double solve(const Matrix & distMatrix, Vector & assignment);

private:
  // Core algorithm phases (preserving original step structure)
  void executeOptimalAssignment();
  void buildAssignmentVector(Vector & assignment);
  double computeAssignmentCost(const Matrix & originalMatrix, const Vector & assignment);

  // Algorithm steps with descriptive names
  void coverColumnsWithStars();
  void checkOptimalityAndProceed();
  void findAndProcessUncoveredZeros();
  void augmentAlternatingPath(int row, int col);
  void updateMatrixValues();

  int findStarInRow(int row) const;
  int findStarInColumn(int col) const;
  int findPrimeInRow(int row) const;

  // Member variables - core algorithm state only
  Matrix distMatrix_;             // Working distance matrix
  BoolMatrix starMatrix_;         // Starred zeros matrix
  BoolMatrix primeMatrix_;        // Primed zeros matrix
  BoolVector coveredRows_;        // Row coverage flags
  BoolVector coveredColumns_;     // Column coverage flags
                                  //
  int nRows_;                     // Matrix row count (cached)
  int nCols_;                     // Matrix column count (cached)
};
