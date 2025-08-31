///////////////////////////////////////////////////////////////////////////////
// hungarian.hpp: Clean Hungarian Algorithm implementation using Eigen
// Based on Munkres algorithm with clear step-by-step structure
// Each method corresponds directly to classic algorithm steps
// Supports both cost minimization and profit maximization
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <limits>
#include <utility>

#include <Eigen/Dense>


namespace hungarian
{

class Hungarian
{
public:
  // Helper type definitions for better readability
  using MatrixXd = Eigen::MatrixXd;
  using VectorXi = Eigen::VectorXi;
  using BoolMatrix = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
  using BoolVector = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

  Hungarian() = default;
  ~Hungarian() = default;

  /**
   * @brief Solve assignment problem using Hungarian/Munkres algorithm
   * @param matrix Cost matrix (minimize=true) or profit matrix (minimize=false)
   * @param assignment Output vector where assignment[i] = j means worker i assigned to task j
   * @param minimize If true, minimize costs; if false, maximize profits
   * @return Total cost (if minimize=true) or total profit (if minimize=false) of optimal assignment
   * @throws std::invalid_argument for invalid input matrices
   */
  double solve(const MatrixXd & matrix, VectorXi & assignment, bool minimize = true);

private:
  // Algorithm state and flow control

  /**
   * @brief Initialize all algorithm state variables and data structures
   * @param matrix Input matrix to initialize working data from
   * @param minimize Whether we're minimizing (true) or maximizing (false)
   * @details Sets up dimensions, transforms matrix if needed, and initializes all boolean matrices and vectors to zero
   */
  void initializeAlgorithm(const MatrixXd & matrix, bool minimize);

  /**
   * @brief Transform profit matrix to cost matrix for maximization problems
   * @param profitMatrix Original profit matrix
   * @return Transformed cost matrix (max_profit - each element)
   * @details Uses the standard transformation: cost[i][j] = max_profit - profit[i][j]
   */
  MatrixXd transformProfitToCost(const MatrixXd & profitMatrix) const;

  /**
   * @brief Execute the main Hungarian algorithm loop until optimal solution is found
   * @throws std::runtime_error if algorithm fails to converge within maximum iterations
   * @details Performs initial steps 1-3, then loops through steps 4-6 until completion
   */
  void executeMainLoop();

  /**
   * @brief Check if optimal solution has been found
   * @return True if number of covered columns equals minimum matrix dimension
   * @details Solution is optimal when we have covered as many columns as the smaller dimension
   */
  bool isOptimalSolutionFound() const;

  // Classic Hungarian algorithm steps

  /**
   * @brief Step 1: Reduce matrix by subtracting row or column minimums
   * @details For matrices with rows <= cols: subtract minimum from each row
   *          For matrices with rows > cols: subtract minimum from each column
   */
  void step1_ReduceMatrix();

  /**
   * @brief Step 2: Find zeros in reduced matrix and star them (one per row/column)
   * @details Stars zeros to create initial assignment, ensuring no two stars share a row or column
   *          Strategy depends on matrix dimensions for optimal performance
   */
  void step2_StarZeros();

  /**
   * @brief Step 3: Cover all columns that contain starred zeros
   * @details Updates column coverage based on current star positions
   */
  void step3_CoverStarredColumns();

  /**
   * @brief Step 4: Find uncovered zero and determine next action
   * @details If uncovered zero found: prime it and either augment path or adjust coverage
   *          If no uncovered zero found: proceed to matrix update (Step 6)
   */
  void step4_FindUncoveredZero();

  /**
   * @brief Step 5: Construct alternating path and update star assignments
   * @param row Row coordinate of the primed zero that starts the augmenting path
   * @param col Column coordinate of the primed zero that starts the augmenting path
   * @details Creates alternating sequence of starred and primed zeros, then updates assignments
   *          Clears all primes and row coverage, returns to Step 3
   */
  void step5_AugmentPath(const int row, const int col);

  /**
   * @brief Step 6: Update working matrix by adding/subtracting minimum uncovered value
   * @details Adds minimum to all covered rows, subtracts minimum from all uncovered columns
   *          Creates new zeros while preserving existing zero structure
   */
  void step6_UpdateMatrix();

  // Result extraction

  /**
   * @brief Build final assignment vector from starred zero positions
   * @param assignment Output vector to populate with assignments
   * @details For each row, finds the starred column (if any) and records the assignment
   *          Unassigned rows get value -1
   */
  void buildAssignmentVector(VectorXi & assignment) const;

  /**
   * @brief Compute total cost/profit of assignment using original matrix
   * @param originalMatrix Original input matrix (before algorithm modifications)
   * @param assignment Assignment vector where assignment[i] = j means worker i -> task j
   * @return Sum of costs/profits for all assignments
   * @throws std::invalid_argument if assignment vector size doesn't match matrix rows
   */
  double computeTotalCost(const MatrixXd & originalMatrix, const VectorXi & assignment) const;

  // Utility methods for matrix operations

  /**
   * @brief Find column index of starred zero in given row
   * @param row Row index to search
   * @return Column index of starred zero, or -1 if no starred zero in row
   * @details Returns first starred zero found (there should be at most one per row)
   */
  int findStarInRow(const int row) const;

  /**
   * @brief Find row index of starred zero in given column
   * @param col Column index to search
   * @return Row index of starred zero, or -1 if no starred zero in column
   * @details Returns first starred zero found (there should be at most one per column)
   */
  int findStarInColumn(const int col) const;

  /**
   * @brief Find column index of primed zero in given row
   * @param row Row index to search
   * @return Column index of primed zero, or -1 if no primed zero in row
   * @details Used during path augmentation to trace alternating sequences
   */
  int findPrimeInRow(const int row) const;

  /**
   * @brief Find coordinates of first uncovered zero in working matrix
   * @return Pair of (row, col) coordinates, or (-1, -1) if no uncovered zero found
   * @details Searches row by row through uncovered positions for zeros (within epsilon tolerance)
   */
  std::pair<int, int> findUncoveredZero() const;

  /**
   * @brief Find minimum value among all uncovered matrix elements
   * @return Minimum uncovered value in working matrix
   * @throws std::runtime_error if no uncovered elements found
   * @details Used in Step 6 to determine adjustment value for matrix update
   */
  double findMinimumUncoveredValue() const;

  // Algorithm state variables
  MatrixXd workingMatrix_;    // Working distance matrix (modified during algorithm)
  BoolMatrix starMatrix_;     // Starred zeros matrix (current assignment)
  BoolMatrix primeMatrix_;    // Primed zeros matrix (temporary markings)
  BoolVector coveredRows_;    // Row coverage flags
  BoolVector coveredColumns_; // Column coverage flags
  int nRows_;                 // Matrix row count (cached for performance)
  int nCols_;                 // Matrix column count (cached for performance)

  // Constants for better maintainability
  static constexpr double EPSILON = std::numeric_limits<double>::epsilon();
};

} // namespace hungarian
