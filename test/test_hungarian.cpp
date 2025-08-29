#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "hungarian/hungarian.hpp"


class HungarianTest : public ::testing::Test
{
protected:
  Hungarian solver;

  // Helper function to print assignment for debugging
  void printAssignment(const Hungarian::VectorXi & assignment, double cost) {
    std::cout << "Assignment: ";
    for (int i = 0; i < assignment.size(); ++i) {
      std::cout << "(" << i << "->" << assignment(i) << ") ";
    }
    std::cout << " Cost: " << cost << std::endl;
  }
};

TEST_F(HungarianTest, BasicSquareMatrix3x3_Test1)
{
  Hungarian::MatrixXd matrix(3, 3);
  matrix << 25.0, 40.0, 35.0,
            40.0, 60.0, 35.0,
            20.0, 40.0, 25.0;
  Hungarian::VectorXi assignment;

  double cost = solver.solve(matrix, assignment);

  // Verify assignment is valid
  EXPECT_EQ(assignment.size(), 3.0);
  EXPECT_GE(assignment(0), -1);
  EXPECT_GE(assignment(1), -1);
  EXPECT_GE(assignment(2), -1);

  // Expected optimal cost for this matrix should be 95 (0->1, 1->2, 2->0)
  EXPECT_DOUBLE_EQ(cost, 95.0);

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, BasicSquareMatrix3x3_Test2)
{
  Hungarian::MatrixXd matrix(3, 3);
  matrix << 64.0, 18.0, 75.0,
            97.0, 60.0, 24.0,
            87.0, 63.0, 15.0;
  Hungarian::VectorXi assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 3);

  // Expected optimal cost for this matrix should be 129 (0->1, 1->2, 2->0)
  EXPECT_DOUBLE_EQ(cost, 129.0);

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, BasicSquareMatrix4x4_Test1)
{
  Hungarian::MatrixXd matrix(4, 4);
  matrix << 80, 40, 50, 46,
            40, 70, 20, 25,
            30, 10, 20, 30,
            35, 20, 25, 30;
  Hungarian::VectorXi assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 4);

  // Expected optimal cost should be 111
  EXPECT_DOUBLE_EQ(cost, 111.0);

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, RectangularMatrix5x4)
{
  Hungarian::MatrixXd matrix(5, 4);
  matrix << 10.0, 19.0,  8.0, 15.0,
            10.0, 18.0,  7.0, 17.0,
            13.0, 16.0,  9.0, 14.0,
            12.0, 19.0,  8.0, 18.0,
            14.0, 17.0, 10.0, 19.0;
  Hungarian::VectorXi assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 5);

  // Expected optimal cost should be 48
  EXPECT_DOUBLE_EQ(cost, 48.0);

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, LargeMatrix20x8)
{
  Hungarian::MatrixXd matrix(20, 8);
  matrix << 85.0, 12.0, 36.0, 83.0, 50.0, 96.0, 12.0,  1.0,
            84.0, 35.0, 16.0, 17.0, 40.0, 94.0, 16.0, 52.0,
            14.0, 16.0,  8.0, 53.0, 14.0, 12.0, 70.0, 50.0,
            73.0, 83.0, 19.0, 44.0, 83.0, 66.0, 71.0, 18.0,
            36.0, 45.0, 29.0,  4.0, 61.0, 15.0, 70.0, 47.0,
             7.0, 14.0, 11.0, 69.0, 57.0, 32.0, 37.0, 81.0,
             9.0, 65.0, 38.0, 74.0, 87.0, 51.0, 86.0, 52.0,
            52.0, 40.0, 56.0, 10.0, 42.0,  2.0, 26.0, 36.0,
            85.0, 86.0, 36.0, 90.0, 49.0, 89.0, 41.0, 74.0,
            40.0, 67.0,  2.0, 70.0, 18.0,  5.0, 94.0, 43.0,
            85.0, 12.0, 36.0, 83.0, 50.0, 96.0, 12.0,  1.0,
            84.0, 35.0, 16.0, 17.0, 40.0, 94.0, 16.0, 52.0,
            14.0, 16.0,  8.0, 53.0, 14.0, 12.0, 70.0, 50.0,
            73.0, 83.0, 19.0, 44.0, 83.0, 66.0, 71.0, 18.0,
            36.0, 45.0, 29.0,  4.0, 61.0, 15.0, 70.0, 47.0,
             7.0, 14.0, 11.0, 69.0, 57.0, 32.0, 37.0, 81.0,
             9.0, 65.0, 38.0, 74.0, 87.0, 51.0, 86.0, 52.0,
            52.0, 40.0, 56.0, 10.0, 42.0,  2.0, 26.0, 36.0,
            85.0, 86.0, 36.0, 90.0, 49.0, 89.0, 41.0, 74.0,
            40.0, 67.0,  2.0, 70.0, 18.0,  5.0, 94.0, 43.0;

  Hungarian::VectorXi assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 20);

  // Verify all assignments are valid (within column bounds or -1 for unassigned)
  for (int i = 0; i < assignment.size(); ++i) {
    EXPECT_TRUE(assignment(i) == -1 || (assignment(i) >= 0 && assignment(i) < 8));
  }

  // Verify no two rows are assigned to the same column
  std::vector<bool> usedColumns(8, false);
  for (int i = 0; i < assignment.size(); ++i) {
    if (assignment(i) >= 0) {
      EXPECT_FALSE(usedColumns[assignment(i)]) << "Column " << assignment(i) << " assigned multiple times";
      usedColumns[assignment(i)] = true;
    }
  }

  std::cout << "Large matrix optimal cost: " << cost << std::endl;
  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, SingleElementMatrix)
{
  Hungarian::MatrixXd matrix(1, 1);
  matrix << 42.0;
  Hungarian::VectorXi assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 1);
  EXPECT_EQ(assignment(0), 0);
  EXPECT_DOUBLE_EQ(cost, 42.0);
}

TEST_F(HungarianTest, ZeroMatrix)
{
  Hungarian::MatrixXd matrix(3, 3);
  matrix << 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0;
  Hungarian::VectorXi assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 3);
  EXPECT_DOUBLE_EQ(cost, 0.0);

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, IdentityMatrix)
{
  Hungarian::MatrixXd matrix(3, 3);
  matrix << 1.0, 2.0, 3.0,
            2.0, 1.0, 3.0,
            3.0, 2.0, 1.0;
  Hungarian::VectorXi assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 3);
  EXPECT_DOUBLE_EQ(cost, 3.0);  // Should assign diagonal elements

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, NegativeValuesThrowException)
{
  Hungarian::MatrixXd matrix(3, 3);
  matrix << 1.0, -2.0, 3.0,
            2.0,  1.0, 3.0,
            3.0,  2.0, 1.0;
  Hungarian::VectorXi assignment;

  EXPECT_THROW(solver.solve(matrix, assignment), std::invalid_argument);
}

TEST_F(HungarianTest, RectangularMatrix4x5)
{
  // More columns than rows
  Hungarian::MatrixXd matrix(4, 5);
  matrix << 10.0, 19.0, 8.0, 15.0, 12.0,
            10.0, 18.0, 7.0, 17.0, 11.0,
            13.0, 16.0, 9.0, 14.0, 13.0,
            12.0, 19.0, 8.0, 18.0, 10.0;
  Hungarian::VectorXi assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 4);

  // Verify all assignments are valid
  for (int i = 0; i < assignment.size(); ++i) {
    if (assignment(i) >= 0) {
      EXPECT_LT(assignment(i), 5);
    }
  }

  printAssignment(assignment, cost);
}

// Performance test for larger matrices
TEST_F(HungarianTest, PerformanceTest)
{
  const int size = 50;
  Hungarian::MatrixXd matrix(size, size);

  // Fill with random-like values
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix(i, j) = (i * 13 + j * 7) % 100 + 1;  // Deterministic "random" values
    }
  }
  Hungarian::VectorXi assignment;

  auto start = std::chrono::high_resolution_clock::now();
  double cost = solver.solve(matrix, assignment);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  EXPECT_EQ(assignment.size(), size);
  std::cout << "Performance test (50x50): " << cost << " in " << duration.count() << "ms" << std::endl;
}
