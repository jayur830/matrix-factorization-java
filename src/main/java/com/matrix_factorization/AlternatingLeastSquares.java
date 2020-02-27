package com.matrix_factorization;

public interface AlternatingLeastSquares<T> {
    public void optimizeUser(T x, T y, T c, T p, int nUsers, int nFactor, double lambda);
    public void optimizeItem(T x, T y, T c, T p, int nItems, int nFactor, double lambda);
    public double[] loss(T x, T y, T c, T p, double lambda);
}
