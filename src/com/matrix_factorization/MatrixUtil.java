package com.matrix_factorization;

import java.util.Arrays;

public class MatrixUtil {
    public static double[][] transpose(double[][] src) {
        double[][] dst = new double[src[0].length][src.length];
        for (int i = 0; i < src.length; ++i)
            for (int j = 0; j < src[i].length; ++j)
                dst[j][i] = src[i][j];
        return dst;
    }

    public static double[][] fill(int rows, int cols, double value) {
        double[][] matrix = new double[rows][cols];
        for (double[] mat : matrix) Arrays.fill(mat, value);
        return matrix;
    }

    public static double[][] sum(double[][] a, double[][] b) {
        double[][] matrix = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i)
            for (int j = 0; j < a[i].length; ++j)
                matrix[i][j] = a[i][j] + b[i][j];
        return matrix;
    }

    public static double[][] subtract(double[][] a, double[][] b) {
        double[][] matrix = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i)
            for (int j = 0; j < a[i].length; ++j)
                matrix[i][j] = a[i][j] - b[i][j];
        return matrix;
    }

    public static double[][] product(double a, double[][] b) {
        double[][] dst = new double[b.length][b[0].length];
        for (int i = 0; i < dst.length; ++i)
            for (int j = 0; j < dst[i].length; ++j)
                dst[i][j] = a * b[i][j];
        return dst;
    }

    public static double[][] product(double[][] a, double[][] b) {
        double[][] dst = new double[a.length][b[0].length];
        for (int i = 0; i < dst.length; ++i)
            for (int j = 0; j < dst[i].length; ++j)
                for (int k = 0; k < a[0].length; ++k)
                    dst[i][j] += a[i][k] * b[k][j];
        return dst;
    }

    public static double dot(double[] a, double[] b) {
        double total = 0;
        for (int i = 0; i < a.length; ++i)
            total += a[i] * b[i];
        return total;
    }

    public static double[] multiple(double[][] a, double[] b) {
        double[] vector = new double[a.length];
        for (int i = 0; i < vector.length; ++i)
            vector[i] = dot(a[i], b);
        return vector;
    }

    public static double[][] inverse(double[][] matrix) {
        double[][] src = matrix.clone(), dst = identity(src.length);
        for (int j = 0; j < src[0].length; ++j)
            for (int i = j, count = 0; count < src.length; i = i != src.length - 1 ? i + 1 : 0, ++count) {
                double factor = src[i][j];
                if (i == j)
                    for (int k = 0; k < src[i].length; ++k) {
                        src[i][k] /= factor;
                        dst[i][k] /= factor;
                    }
                else
                    for (int k = 0; k < src[i].length; ++k) {
                        src[i][k] += -factor * src[j][k];
                        dst[i][k] += -factor * dst[j][k];
                    }
            }
        return dst;
    }

    public static double[][] identity(int length) {
        return diagonal(length, length, 1);
    }

    public static double[][] diagonal(int rows, int cols, double value) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                matrix[i][j] = i == j ? value : 0;
        return matrix;
    }

    public static double[][] diagonal(double[][] mat) {
        double[][] matrix = new double[mat.length][mat[0].length];
        for (int i = 0; i < mat.length; ++i)
            for (int j = 0; j < mat[0].length; ++j)
                matrix[i][j] = i == j ? mat[i][j] : 0;
        return matrix;
    }

    public static double[][] diagonal(double[] vector) {
        double[][] matrix = new double[vector.length][vector.length];
        for (int i = 0; i < matrix.length; ++i)
            for (int j = 0; j < matrix[i].length; ++j)
                matrix[i][j] = i == j ? vector[i] : 0;
        return matrix;
    }

    public static double[][] square(double[][] matrix) {
        double[][] mat = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; ++i)
            for (int j = 0; j < matrix[i].length; ++j)
                mat[i][j] = matrix[i][j] * matrix[i][j];
        return mat;
    }

    public static double sum(double[][] matrix) {
        double total = 0;
        for (double[] mat : matrix)
            for (double m : mat) total += m;
        return total;
    }

    public static double max(double[][] matrix) {
        double max = Double.NEGATIVE_INFINITY;
        for (double[] mat : matrix)
            for (double m : mat) max = Math.max(max, m);
        return max;
    }

    public static double min(double[][] matrix) {
        double min = Double.POSITIVE_INFINITY;
        for (double[] mat : matrix)
            for (double m : mat) min = Math.min(min, m);
        return min;
    }

    public static double[][] norm(double[][] matrix) {
        double min = min(matrix), max = max(matrix);
        return subtract(product(1 / (max - min), matrix), product(1 / (max - min), fill(matrix.length, matrix[0].length, min)));
    }

    public static double[][] norm(double[][] matrix, double min, double max) {
        return sum(product(max - min, norm(matrix)), fill(matrix.length, matrix[0].length, min));
    }

    public static double[][] elementWise(double[][] a, double[][] b) {
        double[][] matrix = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i)
            for (int j = 0; j < a[i].length; ++j)
                matrix[i][j] = a[i][j] * b[i][j];
        return matrix;
    }

    public static double[] col(double[][] mat, int col) {
        double[] vector = new double[mat.length];
        for (int i = 0; i < vector.length; ++i)
            vector[i] = mat[i][col];
        return vector;
    }

    public static double[] solve(double[][] x, double[] y) {
        return transpose(product(inverse(x), transpose(new double[][] { y })))[0];
    }

    public static void print(double[][] matrix) {
        for (double[] mat : matrix) {
            for (double m : mat)
                System.out.printf("%3.2f ", m);
            System.out.println();
        }
    }
}
