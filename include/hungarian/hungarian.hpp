///////////////////////////////////////////////////////////////////////////////
// hungarian.hpp: Clean Hungarian Algorithm implementation using Eigen
// Based on Munkres algorithm with clear step-by-step structure
// Each method corresponds directly to classic algorithm steps
///////////////////////////////////////////////////////////////////////////////

#pragma once

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

  /**
   * @brief Solve assignment problem using Hungarian/Munkres algorithm
   * @param costMatrix Cost matrix (rows=workers, cols=tasks)
   * @param assignment Output vector where assignment[i] = j means worker i assigned to task j
   * @return Total cost of optimal assignment
   * @throws std::invalid_argument for invalid input matrices
   */
  double solve(const Matrix & costMatrix, Vector & assignment);

private:
  // Algorithm state and flow control
  void initializeAlgorithm(const Matrix & costMatrix);
  void executeMainLoop();
  bool isOptimalSolutionFound() const;

  // Classic Hungarian algorithm steps
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
