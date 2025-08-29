///////////////////////////////////////////////////////////////////////////////
// hungarian.hpp: Clean Hungarian Algorithm implementation using Eigen
// Based on Munkres algorithm with clear step-by-step structure
// Each method corresponds directly to classic algorithm steps
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
  // Algorithm state and flow control
  void initializeAlgorithm(const Matrix & distMatrix);
  void executeMainLoop();
  bool isOptimalSolutionFound() const;

  // Classic Munkres algorithm steps (clear 1-to-1 mapping)
  void step1_ReduceMatrix();
  void step2_StarZeros();
  void step3_CoverStarredColumns();
  void step4_FindUncoveredZero();
  void step5_AugmentPath(const int row, const int col);
  void step6_UpdateMatrix();

  // Result extraction
  void buildAssignmentVector(Vector & assignment) const;
  double computeTotalCost(const Matrix & originalMatrix, const Vector & assignment) const;

  // Utility methods for matrix operations
  int findStarInRow(const int row) const;
  int findStarInColumn(const int col) const;
  int findPrimeInRow(const int row) const;
  std::pair<int, int> findUncoveredZero() const;
  double findMinimumUncoveredValue() const;

  // Algorithm state variables
  Matrix workingMatrix_;      // Working distance matrix
  BoolMatrix starMatrix_;     // Starred zeros matrix
  BoolMatrix primeMatrix_;    // Primed zeros matrix
  BoolVector coveredRows_;    // Row coverage flags
  BoolVector coveredColumns_; // Column coverage flags
  int nRows_;                 // Matrix row count (cached)
  int nCols_;                 // Matrix column count (cached)

  // Constants for better maintainability
  static constexpr double EPSILON = std::numeric_limits<double>::epsilon();
};
