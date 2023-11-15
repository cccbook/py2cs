import unittest

from strassen import matrix_addition, matrix_subtraction, strassen


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.matrix = [
            [1, 2],
            [1, 2]
        ]

        self.matrix_2 = [
            [5, 6],
            [5, 6]
        ]

        self.big_m_1 = [
            [10, 9, 4, 3],
            [8, 3, 4, 1],
            [93, 1, 9, 3],
            [2, 2, 7, 6]
        ]

        self.big_m_2 = [
            [4, 5, 3, 5],
            [4, 1, 2, 1],
            [9, 8, 3, 5],
            [6, 3, 7, 9]
        ]

    def test_matrix_addition(self):
        expected_matrix = [[6, 8], [6, 8]]
        self.assertEqual(matrix_addition(self.matrix, self.matrix_2), expected_matrix)

    def test_matrix_subtraction(self):
        expected_matrix = [[-4, -4], [-4, -4]]
        self.assertEqual(matrix_subtraction(self.matrix, self.matrix_2), expected_matrix)

    def test_strassen_multiplication(self):
        expected_matrix = [[130, 100, 81, 106],
                           [86, 78, 49, 72],
                           [475, 547, 329, 538],
                           [115, 86, 73, 101]]
        self.assertEqual(strassen(self.big_m_1, self.big_m_2), expected_matrix)

if __name__ == '__main__':
    unittest.main()