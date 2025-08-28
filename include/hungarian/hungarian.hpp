///////////////////////////////////////////////////////////////////////////////
// HungarianEigen.h: Modern Hungarian Algorithm implementation using Eigen
// Based on Munkres algorithm for solving assignment problems
// Modernized version with Eigen library for better performance and readability
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>
#include <limits>

#include <Eigen/Dense>


class Hungarian
{
public:
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXi;
  using BoolMatrix = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
  using BoolVector = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

  Hungarian() = default;
  ~Hungarian() = default;

  double solve(const Matrix& costMatrix, Vector& assignment);

private:
  // Core algorithm steps
  void initializeMatrix(Matrix& distMatrix);
  void findInitialAssignment();
  bool findOptimalAssignment();
  void augmentPath(int row, int col);
  void updateMatrix();
  double computeCost(const Matrix& originalMatrix, const Vector& assignment);

  // Helper functions
  bool findUncoveredZero(int& row, int& col);
  bool findStarInRow(int row, int& col);
  bool findStarInColumn(int col, int& row);
  bool findPrimeInRow(int row, int& col);
  double findMinUncoveredValue();

  // Algorithm state
  Matrix distMatrix_;
  BoolMatrix starMatrix_;
  BoolMatrix primeMatrix_;
  BoolVector coveredRows_;
  BoolVector coveredColumns_;
  int nRows_, nCols_, minDim_;
};
