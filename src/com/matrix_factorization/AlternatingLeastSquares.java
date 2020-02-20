package com.matrix_factorization;

public class AlternatingLeastSquares {

    public AlternatingLeastSquares() {}

    /**
     * @param x: nUsers * factor
     * @param y: nItems * factor
     * @param c: nUsers * nItems
     * @param p: nUsers * nItems
     * */
    public void optimizeUser(double[][] x, double[][] y, double[][] c, double[][] p, int nUsers, int nFactor, double lambda) {
        // yt: nFactor * nItems
        double[][] yt = MatrixUtil.transpose(y);
        for (int u = 0; u < nUsers; ++u) {
            // cu: nItems * nItems
            double[][] cu = MatrixUtil.diagonal(c[u]);
            // yt_cu: nFactor * nItems
            double[][] yt_cu = MatrixUtil.product(yt, cu);
            // yt_cu_y: nFactor * nFactor
            double[][] yt_cu_y = MatrixUtil.product(yt_cu, y);
            // li: nFactor * nFactor
            double[][] li = MatrixUtil.diagonal(nFactor, nFactor, lambda);
            // yt_cu_pu: 1 * nFactor
            double[] yt_cu_pu = MatrixUtil.multiple(yt_cu, p[u]);
            x[u] = MatrixUtil.solve(MatrixUtil.sum(yt_cu_y, li), yt_cu_pu);
        }
    }

    public void optimizeItem(double[][] x, double[][] y, double[][] c, double[][] p, int nItems, int nFactor, double lambda) {
        // yt: nFactor * nUsers
        double[][] xt = MatrixUtil.transpose(x);
        for (int i = 0; i < nItems; ++i) {
            // ci: nUsers * nUsers
            double[][] ci = MatrixUtil.diagonal(MatrixUtil.col(c, i));
            // xt_ci: nFactor * nUsers
            double[][] xt_ci = MatrixUtil.product(xt, ci);
            // xt_ci_x: nFactor * nFactor
            double[][] xt_ci_x = MatrixUtil.product(xt_ci, x);
            // li: nFactor * nFactor
            double[][] li = MatrixUtil.diagonal(nFactor, nFactor, lambda);
            // yt_cu_pu: 1 * nFactor
            double[] xt_ci_pi = MatrixUtil.multiple(xt_ci, MatrixUtil.col(p, i));
            y[i] = MatrixUtil.solve(MatrixUtil.sum(xt_ci_x, li), xt_ci_pi);
        }
    }

    public double[] loss(double[][] x, double[][] y, double[][] c, double[][] p, double lambda) {
        double[][] predict = MatrixUtil.product(x, MatrixUtil.transpose(y));
        double[][] predictError = MatrixUtil.square(MatrixUtil.subtract(p, predict));
        double confidenceError = MatrixUtil.sum(MatrixUtil.elementWise(c, predictError));
        double regularization = lambda * (MatrixUtil.sum(MatrixUtil.square(x)) + MatrixUtil.sum(MatrixUtil.square(y)));
        double totalLoss = confidenceError + regularization;
        return new double[] { MatrixUtil.sum(predictError), confidenceError, regularization, totalLoss };
    }
}
