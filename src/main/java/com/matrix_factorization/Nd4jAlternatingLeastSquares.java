package com.matrix_factorization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Nd4jAlternatingLeastSquares implements AlternatingLeastSquares<INDArray> {
    /**
     * @param x: nUsers * factor
     * @param y: nItems * factor
     * @param c: nUsers * nItems
     * @param p: nUsers * nItems
     * */
    public void optimizeUser(INDArray x, INDArray y, INDArray c, INDArray p, int nUsers, int nFactor, double lambda) {
        // yt: nFactor * nItems
        INDArray yt = y.transpose();
        for (int u = 0; u < nUsers; ++u) {
            // cu: nItems * nItems
            INDArray cu = Nd4j.diag(c.getRow(u));
            // yt_cu: nFactor * nItems
            INDArray yt_cu = Nd4j.matmul(yt, cu);
            // yt_cu_y: nFactor * nFactor
            INDArray yt_cu_y = Nd4j.matmul(yt_cu, y);
            // li: nFactor * nFactor
            INDArray li = Nd4j.eye(nFactor).mul(lambda);
            // yt_cu_pu: nFactor * 1
            INDArray yt_cu_pu = Nd4j.matmul(yt_cu, p.getRows(u).transpose());

            x.putRow(u, Nd4j.matmul(InvertMatrix.invert(yt_cu_y.add(li), false), yt_cu_pu).transpose());
        }
    }

    public void optimizeItem(INDArray x, INDArray y, INDArray c, INDArray p, int nItems, int nFactor, double lambda) {
        // yt: nFactor * nUsers
        INDArray xt = x.transpose();
        for (int i = 0; i < nItems; ++i) {
            // ci: nUsers * nUsers
            INDArray ci = Nd4j.diag(c.getColumn(i));
            // xt_ci: nFactor * nUsers
            INDArray xt_ci = Nd4j.matmul(xt, ci);
            // xt_ci_x: nFactor * nFactor
            INDArray xt_ci_x = Nd4j.matmul(xt_ci, x);
            // li: nFactor * nFactor
            INDArray li = Nd4j.eye(nFactor).mul(lambda);
            // xt_ci_pi: nFactor * 1
            INDArray xt_ci_pi = Nd4j.matmul(xt_ci, p.getColumns(i));

            y.putRow(i, Nd4j.matmul(InvertMatrix.invert(xt_ci_x.add(li), false), xt_ci_pi).transpose());
        }
    }

    public double[] loss(INDArray x, INDArray y, INDArray c, INDArray p, double lambda) {
        INDArray predict = Nd4j.matmul(x, y.transpose());
        INDArray predictError = Transforms.pow(p.sub(predict), 2);
        double confidenceError = c.mul(predictError).sumNumber().doubleValue();
        double regularization = lambda * (Transforms.pow(x, 2).sumNumber().doubleValue() + Transforms.pow(y, 2).sumNumber().doubleValue());
        double totalLoss = confidenceError + regularization;
        return new double[] { predictError.sumNumber().doubleValue(), confidenceError, regularization, totalLoss };
    }
}
