#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "hungarian/hungarian.hpp"


class HungarianTest : public ::testing::Test
{
protected:
  Hungarian solver;

  // Helper function to print assignment for debugging
  void printAssignment(const Hungarian::Vector& assignment, double cost) {
    std::cout << "Assignment: ";
    for (int i = 0; i < assignment.size(); ++i) {
      std::cout << "(" << i << "->" << assignment(i) << ") ";
    }
    std::cout << " Cost: " << cost << std::endl;
  }
};

TEST_F(HungarianTest, BasicSquareMatrix3x3_Test1)
{
  Hungarian::Matrix matrix(3, 3);
  matrix << 25.0, 40.0, 35.0,
            40.0, 60.0, 35.0,
            20.0, 40.0, 25.0;
  Hungarian::Vector assignment;

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
  Hungarian::Matrix matrix(3, 3);
  matrix << 64.0, 18.0, 75.0,
            97.0, 60.0, 24.0,
            87.0, 63.0, 15.0;
  Hungarian::Vector assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 3);

  // Expected optimal cost for this matrix should be 129 (0->1, 1->2, 2->0)
  EXPECT_DOUBLE_EQ(cost, 129.0);

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, BasicSquareMatrix4x4_Test1)
{
  Hungarian::Matrix matrix(4, 4);
  matrix << 80, 40, 50, 46,
            40, 70, 20, 25,
            30, 10, 20, 30,
            35, 20, 25, 30;
  Hungarian::Vector assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 4);

  // Expected optimal cost should be 111
  EXPECT_DOUBLE_EQ(cost, 111.0);

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, RectangularMatrix5x4)
{
  Hungarian::Matrix matrix(5, 4);
  matrix << 10, 19, 8, 15,
            10, 18, 7, 17,
            13, 16, 9, 14,
            12, 19, 8, 18,
            14, 17, 10, 19;
  Hungarian::Vector assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 5);

  // Expected optimal cost should be 48
  EXPECT_DOUBLE_EQ(cost, 48.0);

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, LargeMatrix20x8)
{
  Hungarian::Matrix matrix(20, 8);
  matrix << 85, 12, 36, 83, 50, 96, 12,  1,
            84, 35, 16, 17, 40, 94, 16, 52,
            14, 16,  8, 53, 14, 12, 70, 50,
            73, 83, 19, 44, 83, 66, 71, 18,
            36, 45, 29,  4, 61, 15, 70, 47,
             7, 14, 11, 69, 57, 32, 37, 81,
             9, 65, 38, 74, 87, 51, 86, 52,
            52, 40, 56, 10, 42,  2, 26, 36,
            85, 86, 36, 90, 49, 89, 41, 74,
            40, 67,  2, 70, 18,  5, 94, 43,
            85, 12, 36, 83, 50, 96, 12,  1,
            84, 35, 16, 17, 40, 94, 16, 52,
            14, 16,  8, 53, 14, 12, 70, 50,
            73, 83, 19, 44, 83, 66, 71, 18,
            36, 45, 29,  4, 61, 15, 70, 47,
             7, 14, 11, 69, 57, 32, 37, 81,
             9, 65, 38, 74, 87, 51, 86, 52,
            52, 40, 56, 10, 42,  2, 26, 36,
            85, 86, 36, 90, 49, 89, 41, 74,
            40, 67,  2, 70, 18,  5, 94, 43;

  Hungarian::Vector assignment;

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
  Hungarian::Matrix matrix(1, 1);
  matrix << 42;
  Hungarian::Vector assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 1);
  EXPECT_EQ(assignment(0), 0);
  EXPECT_DOUBLE_EQ(cost, 42.0);
}

TEST_F(HungarianTest, ZeroMatrix)
{
  Hungarian::Matrix matrix(3, 3);
  matrix << 0, 0, 0,
            0, 0, 0,
            0, 0, 0;
  Hungarian::Vector assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 3);
  EXPECT_DOUBLE_EQ(cost, 0.0);

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, IdentityMatrix)
{
  Hungarian::Matrix matrix(3, 3);
  matrix << 1, 2, 3,
            2, 1, 3,
            3, 2, 1;
  Hungarian::Vector assignment;

  double cost = solver.solve(matrix, assignment);

  EXPECT_EQ(assignment.size(), 3);
  EXPECT_DOUBLE_EQ(cost, 3.0);  // Should assign diagonal elements

  printAssignment(assignment, cost);
}

TEST_F(HungarianTest, NegativeValuesThrowException)
{
  Hungarian::Matrix matrix(3, 3);
  matrix << 1, -2, 3,
            2, 1, 3,
            3, 2, 1;
  Hungarian::Vector assignment;

  EXPECT_THROW(solver.solve(matrix, assignment), std::invalid_argument);
}

TEST_F(HungarianTest, RectangularMatrix4x5)
{
  // More columns than rows
  Hungarian::Matrix matrix(4, 5);
  matrix << 10, 19, 8, 15, 12,
            10, 18, 7, 17, 11,
            13, 16, 9, 14, 13,
            12, 19, 8, 18, 10;
  Hungarian::Vector assignment;

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
  Hungarian::Matrix matrix(size, size);

  // Fill with random-like values
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix(i, j) = (i * 13 + j * 7) % 100 + 1;  // Deterministic "random" values
    }
  }
  Hungarian::Vector assignment;

  auto start = std::chrono::high_resolution_clock::now();
  double cost = solver.solve(matrix, assignment);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  EXPECT_EQ(assignment.size(), size);
  std::cout << "Performance test (50x50): " << cost << " in " << duration.count() << "ms" << std::endl;
}
