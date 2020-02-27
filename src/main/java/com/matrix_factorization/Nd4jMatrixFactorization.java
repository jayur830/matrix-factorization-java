package com.matrix_factorization;

import org.apache.log4j.BasicConfigurator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.Arrays;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class Nd4jMatrixFactorization implements MatrixFactorization {
    static {
        BasicConfigurator.configure();
    }

    private int nUsers, nItems, nFactor;
    private double lambda, alpha;
    private INDArray r, x, y, p, c;

    private boolean printLog = true;

    private Nd4jAlternatingLeastSquares als;

    public Nd4jMatrixFactorization(double[][] r, int nFactor, double lambda, double alpha) {
        this.lambda = lambda; this.alpha = alpha;
        this.x = Nd4j.rand(this.nUsers = (this.r = Nd4j.create(r)).rows(), this.nFactor = nFactor).mul(0.01);
        this.y = Nd4j.rand(this.nItems = this.r.columns(), this.nFactor).mul(0.01);
        init();

        this.als = new Nd4jAlternatingLeastSquares();
    }

    public Nd4jMatrixFactorization(double[][] r, int nFactor, double lambda, double alpha, boolean printLog) {
        this(r, nFactor, lambda, alpha);
        setPrintLog(printLog);
    }

    public Nd4jMatrixFactorization(int nUsers, int nItems, int nFactor, double lambda, double alpha) {
        this.r = Nd4j.zeros(this.nUsers = nUsers, this.nItems = nItems);

        this.lambda = lambda; this.alpha = alpha;
        this.x = Nd4j.rand(this.nUsers, this.nFactor = nFactor).mul(0.01);
        this.y = Nd4j.rand(this.nItems, this.nFactor).mul(0.01);
        init();

        this.als = new Nd4jAlternatingLeastSquares();
    }

    public Nd4jMatrixFactorization(int nUsers, int nItems, int nFactor, double lambda, double alpha, boolean printLog) {
        this(nUsers, nItems, nFactor, lambda, alpha);
        setPrintLog(printLog);
    }

    public Nd4jMatrixFactorization(String modelPath) {
        loadModel(modelPath);
        this.als = new Nd4jAlternatingLeastSquares();
    }

    @Override
    public void setPrintLog(boolean printLog) {
        this.printLog = printLog;
    }

    @Override
    public void fit(int stepSize) {
        for (int step = 1; step <= stepSize; ++step) {
            this.als.optimizeUser(this.x, this.y, this.c, this.p, this.nUsers, this.nFactor, this.lambda);
            this.als.optimizeItem(this.x, this.y, this.c, this.p, this.nItems, this.nFactor, this.lambda);

            if (this.printLog) {
                double[] loss = this.als.loss(this.x, this.y, this.c, this.p, this.lambda);
                System.out.println("------------------------------Step " + step + "----------------------------");
                System.out.println("predict error: " + loss[0]);
                System.out.println("confidence error: " + loss[1]);
                System.out.println("regularization: " + loss[2]);
                System.out.println("total loss: " + loss[3]);
            }
        }
    }

    @Override
    public double[][] predict() {
        return Nd4j.matmul(this.x, this.y.transpose()).toDoubleMatrix();
    }

    @Override
    public double[][] getR() {
        return this.r.toDoubleMatrix();
    }

    @Override
    public void put(int user, int item, double rating) {
        this.r.put(user, item, rating);
        this.p.put(user, item, rating > 0 ? 1 : 0);
        this.c.put(user, item, 1 + this.alpha * rating);
    }

    @Override
    public void addUser(double[] user) {
        addUsers(new double[][] { user });
    }

    @Override
    public void addUsers(double[][] _users) {
        this.nUsers += _users.length;
        this.r = Nd4j.concat(0, this.r, Nd4j.create(_users));
        this.x = Nd4j.concat(0, this.x, Nd4j.rand(_users.length, this.nFactor).mul(0.01));

        init();
    }

    @Override
    public void addEmptyUser() {
        double[] user = new double[this.nItems];
        Arrays.fill(user, 0);
        addUser(user);
    }

    @Override
    public void addEmptyUsers(int nUsers) {
        double[][] users = new double[nUsers][this.nItems];
        for (double[] user : users) Arrays.fill(user, 0);
        addUsers(users);
    }

    @Override
    public void addEmptyItem() {
        addEmptyItems(1);
    }

    @Override
    public void addEmptyItems(int nItems) {
        this.nItems += nItems;
        this.r = Nd4j.concat(0, this.r, Nd4j.zeros(this.nUsers, nItems));
        this.y = Nd4j.concat(0, this.y, Nd4j.rand(nItems, this.nFactor).mul(0.01));

        init();
    }

    private void init() {
        this.c = Nd4j.create(this.r.rows(), this.r.columns());
        this.p = Nd4j.create(this.r.rows(), this.r.columns());
        for (int u = 0; u < this.r.rows(); ++u)
            for (int i = 0; i < this.r.columns(); ++i) {
                this.c.put(u, i, 1 + this.alpha * this.r.getDouble(u, i));
                this.p.put(u, i, this.r.getDouble(u, i) > 0 ? 1 : 0);
            }
    }

    @Override
    public void saveModel(String fileName) {
        if (fileName.contains(".zip")) {
            String[] confs = new String[]{"conf.csv", "r.csv", "x.csv", "y.csv", "c.csv", "p.csv"};
            try {
                for (String conf : confs) {
                    BufferedWriter writer = new BufferedWriter(new FileWriter(conf));
                    if (conf.equals("conf.csv")) {
                        writer.write(this.nUsers + ","
                                + this.nItems + ","
                                + this.nFactor + ","
                                + this.lambda + ","
                                + this.alpha);
                    } else if (conf.equals("r.csv")) {
                        writer.write(this.nUsers + "," + this.nItems);
                        writer.newLine();
                        for (double[] vec : this.r.toDoubleMatrix()) {
                            StringBuilder s = new StringBuilder().append(vec[0]);
                            for (int j = 1; j < vec.length; ++j)
                                s.append(",").append(vec[j]);
                            writer.write(s.toString());
                            writer.newLine();
                        }
                    } else if (conf.equals("x.csv")) {
                        writer.write(this.nUsers + "," + this.nFactor);
                        writer.newLine();
                        for (double[] vec : this.x.toDoubleMatrix()) {
                            StringBuilder s = new StringBuilder().append(vec[0]);
                            for (int j = 1; j < vec.length; ++j)
                                s.append(",").append(vec[j]);
                            writer.write(s.toString());
                            writer.newLine();
                        }
                    } else if (conf.equals("y.csv")) {
                        writer.write(this.nItems + "," + this.nFactor);
                        writer.newLine();
                        for (double[] vec : this.y.toDoubleMatrix()) {
                            StringBuilder s = new StringBuilder().append(vec[0]);
                            for (int j = 1; j < vec.length; ++j)
                                s.append(",").append(vec[j]);
                            writer.write(s.toString());
                            writer.newLine();
                        }
                    } else if (conf.equals("c.csv")) {
                        writer.write(this.nUsers + "," + this.nItems);
                        writer.newLine();
                        for (double[] vec : this.c.toDoubleMatrix()) {
                            StringBuilder s = new StringBuilder().append(vec[0]);
                            for (int j = 1; j < vec.length; ++j)
                                s.append(",").append(vec[j]);
                            writer.write(s.toString());
                            writer.newLine();
                        }
                    } else if (conf.equals("p.csv")) {
                        writer.write(this.nUsers + "," + this.nItems);
                        writer.newLine();
                        for (double[] vec : this.p.toDoubleMatrix()) {
                            StringBuilder s = new StringBuilder().append(vec[0]);
                            for (int j = 1; j < vec.length; ++j)
                                s.append(",").append(vec[j]);
                            writer.write(s.toString());
                            writer.newLine();
                        }
                    }
                    writer.flush();
                    writer.close();
                }

                byte[] buf = new byte[1024];
                ZipOutputStream zip = new ZipOutputStream(new FileOutputStream(fileName));
                for (String conf : confs) {
                    FileInputStream inputStream = new FileInputStream(conf);
                    zip.putNextEntry(new ZipEntry(conf));

                    int length;
                    while ((length = inputStream.read(buf)) > 0)
                        zip.write(buf, 0, length);

                    zip.closeEntry();
                    inputStream.close();
                    new File(conf).delete();
                }
                zip.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void loadModel(String modelPath) {
        if (modelPath.contains(".zip")) {
            try {
                ZipInputStream zip = new ZipInputStream(new FileInputStream(modelPath));
                ZipEntry entry;
                while ((entry = zip.getNextEntry()) != null) {
                    String fileName = entry.getName(), line;
                    FileOutputStream outputStream = new FileOutputStream(fileName);
                    int length;
                    while ((length = zip.read()) != -1)
                        outputStream.write(length);
                    zip.closeEntry();
                    outputStream.flush();
                    outputStream.close();
                    File file = new File(fileName);
                    BufferedReader reader = new BufferedReader(new FileReader(file));
                    String[] conf, shape;
                    if (fileName.equals("conf.csv")) {
                        conf = reader.readLine().split(",");
                        nUsers = Integer.parseInt(conf[0]);
                        nItems = Integer.parseInt(conf[1]);
                        nFactor = Integer.parseInt(conf[2]);
                        lambda = Double.parseDouble(conf[3]);
                        alpha = Double.parseDouble(conf[4]);
                    } else if (fileName.equals("r.csv")) {
                        shape = reader.readLine().split(",");
                        nUsers = Integer.parseInt(shape[0]);
                        nItems = Integer.parseInt(shape[1]);
                        r = Nd4j.create(nUsers, nItems);

                        for (int i = 0; i < r.rows() && (line = reader.readLine()) != null; ++i) {
                            String[] elements = line.split(",");
                            for (int j = 0; j < r.columns(); ++j)
                                r.put(i, j, Double.parseDouble(elements[j]));
                        }
                    } else if (fileName.equals("x.csv")) {
                        shape = reader.readLine().split(",");
                        nUsers = Integer.parseInt(shape[0]);
                        nFactor = Integer.parseInt(shape[1]);
                        x = Nd4j.create(nUsers, nFactor);

                        for (int i = 0; i < x.rows() && (line = reader.readLine()) != null; ++i) {
                            String[] elements = line.split(",");
                            for (int j = 0; j < x.columns(); ++j)
                                x.put(i, j, Double.parseDouble(elements[j]));
                        }
                    } else if (fileName.equals("y.csv")) {
                        shape = reader.readLine().split(",");
                        nItems = Integer.parseInt(shape[0]);
                        nFactor = Integer.parseInt(shape[1]);
                        y = Nd4j.create(nItems, nFactor);

                        for (int i = 0; i < y.rows() && (line = reader.readLine()) != null; ++i) {
                            String[] elements = line.split(",");
                            for (int j = 0; j < y.columns(); ++j)
                                y.put(i, j, Double.parseDouble(elements[j]));
                        }
                    } else if (fileName.equals("c.csv")) {
                        shape = reader.readLine().split(",");
                        nUsers = Integer.parseInt(shape[0]);
                        nItems = Integer.parseInt(shape[1]);
                        c = Nd4j.create(nUsers, nItems);

                        for (int i = 0; i < c.rows() && (line = reader.readLine()) != null; ++i) {
                            String[] elements = line.split(",");
                            for (int j = 0; j < c.columns(); ++j)
                                c.put(i, j, Double.parseDouble(elements[j]));
                        }
                    } else if (fileName.equals("p.csv")) {
                        shape = reader.readLine().split(",");
                        nUsers = Integer.parseInt(shape[0]);
                        nItems = Integer.parseInt(shape[1]);
                        p = Nd4j.create(nUsers, nItems);

                        for (int i = 0; i < p.rows() && (line = reader.readLine()) != null; ++i) {
                            String[] elements = line.split(",");
                            for (int j = 0; j < p.columns(); ++j)
                                p.put(i, j, Double.parseDouble(elements[j]));
                        }
                    }
                    reader.close();
                    file.delete();
                }
                zip.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
