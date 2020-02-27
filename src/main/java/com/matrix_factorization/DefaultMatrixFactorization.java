package com.matrix_factorization;

import org.apache.log4j.BasicConfigurator;

import java.io.*;
import java.util.Arrays;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class DefaultMatrixFactorization implements MatrixFactorization {
    static {
        BasicConfigurator.configure();
    }

    private int nUsers, nItems, nFactor;
    private double lambda, alpha;
    private double[][] r, x, y, p, c;

    private boolean printLog = true;

    private DefaultAlternatingLeastSquares als;

    public DefaultMatrixFactorization(double[][] r, int nFactor, double lambda, double alpha) {
        this.lambda = lambda; this.alpha = alpha;
        initWeights(this.x = new double[this.nUsers = (this.r = r).length][this.nFactor = nFactor]);
        initWeights(this.y = new double[this.nItems = this.r[0].length][this.nFactor]);
        init();

        this.als = new DefaultAlternatingLeastSquares();
    }

    public DefaultMatrixFactorization(double[][] r, int nFactor, double lambda, double alpha, boolean printLog) {
        this(r, nFactor, lambda, alpha);
        setPrintLog(printLog);
    }

    public DefaultMatrixFactorization(int nUsers, int nItems, int nFactor, double lambda, double alpha) {
        this.r = new double[this.nUsers = nUsers][this.nItems = nItems];
        for (double[] mat : this.r) Arrays.fill(mat, 0);

        this.lambda = lambda; this.alpha = alpha;
        initWeights(this.x = new double[this.nUsers][this.nFactor = nFactor]);
        initWeights(this.y = new double[this.nItems][this.nFactor]);
        init();

        this.als = new DefaultAlternatingLeastSquares();
    }

    public DefaultMatrixFactorization(int nUsers, int nItems, int nFactor, double lambda, double alpha, boolean printLog) {
        this(nUsers, nItems, nFactor, lambda, alpha);
        setPrintLog(printLog);
    }

    public DefaultMatrixFactorization(String modelPath) {
        loadModel(modelPath);
        this.als = new DefaultAlternatingLeastSquares();
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
        return MatrixUtil.product(this.x, MatrixUtil.transpose(this.y));
    }

    @Override
    public double[][] getR() {
        return this.r;
    }

    @Override
    public void put(int user, int item, double rating) {
        this.r[user][item] = rating;
        this.p[user][item] = rating > 0 ? 1 : 0;
        this.c[user][item] = 1 + this.alpha * rating;
    }

    @Override
    public void addUser(double[] user) {
        addUsers(new double[][] { user });
    }

    @Override
    public void addUsers(double[][] users) {
        double[][] r = new double[this.nUsers = this.r.length + users.length][this.nItems];
        for (int i = 0; i < this.r.length; ++i)
            System.arraycopy(this.r[i], 0, r[i], 0, this.nItems);
        for (int i = 0; i < users.length; ++i)
            System.arraycopy(users[i], 0, r[this.r.length + i], 0, this.nItems);
        this.r = r;

        double[][] x = new double[this.nUsers][this.nFactor];
        for (int i = 0; i < this.x.length; ++i)
            System.arraycopy(this.x[i], 0, x[i], 0, this.nFactor);
        Random random = new Random();
        for (int i = this.x.length; i < x.length; ++i)
            for (int j = 0; j < x[i].length; ++j)
                x[i][j] = random.nextDouble() * 0.01;
        this.x = x;

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
        double[][] r = new double[this.nUsers][this.nItems + nItems];
        for (int i = 0; i < this.nUsers; ++i) {
            System.arraycopy(this.r[i], 0, r[i], 0, this.nItems);
            for (int j = this.r[i].length; j < r[i].length; ++j) r[i][j] = 0;
        }
        this.r = r;

        double[][] y = new double[this.nItems + nItems][this.nFactor];
        for (int i = 0; i < this.y.length; ++i)
            System.arraycopy(this.y[i], 0, y[i], 0, this.nFactor);
        Random random = new Random();
        for (int i = this.y.length; i < y.length; ++i)
            for (int j = 0; j < y[i].length; ++j)
                y[i][j] = random.nextDouble() * 0.01;
        this.y = y;

        init();
    }

    private void init() {
        this.c = new double[this.r.length][this.r[0].length];
        this.p = new double[this.r.length][this.r[0].length];
        for (int u = 0; u < this.r.length; ++u)
            for (int i = 0; i < this.r[u].length; ++i) {
                this.c[u][i] = 1 + this.alpha * this.r[u][i];
                this.p[u][i] = this.r[u][i] > 0 ? 1 : 0;
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
                        for (double[] vec : this.r) {
                            StringBuilder s = new StringBuilder().append(vec[0]);
                            for (int j = 1; j < vec.length; ++j)
                                s.append(",").append(vec[j]);
                            writer.write(s.toString());
                            writer.newLine();
                        }
                    } else if (conf.equals("x.csv")) {
                        writer.write(this.nUsers + "," + this.nFactor);
                        writer.newLine();
                        for (double[] vec : this.x) {
                            StringBuilder s = new StringBuilder().append(vec[0]);
                            for (int j = 1; j < vec.length; ++j)
                                s.append(",").append(vec[j]);
                            writer.write(s.toString());
                            writer.newLine();
                        }
                    } else if (conf.equals("y.csv")) {
                        writer.write(this.nItems + "," + this.nFactor);
                        writer.newLine();
                        for (double[] vec : this.y) {
                            StringBuilder s = new StringBuilder().append(vec[0]);
                            for (int j = 1; j < vec.length; ++j)
                                s.append(",").append(vec[j]);
                            writer.write(s.toString());
                            writer.newLine();
                        }
                    } else if (conf.equals("c.csv")) {
                        writer.write(this.nUsers + "," + this.nItems);
                        writer.newLine();
                        for (double[] vec : this.c) {
                            StringBuilder s = new StringBuilder().append(vec[0]);
                            for (int j = 1; j < vec.length; ++j)
                                s.append(",").append(vec[j]);
                            writer.write(s.toString());
                            writer.newLine();
                        }
                    } else if (conf.equals("p.csv")) {
                        writer.write(this.nUsers + "," + this.nItems);
                        writer.newLine();
                        for (double[] vec : this.p) {
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
                        r = new double[nUsers][nItems];

                        for (int i = 0; i < r.length && (line = reader.readLine()) != null; ++i) {
                            String[] elements = line.split(",");
                            for (int j = 0; j < r[i].length; ++j)
                                r[i][j] = Double.parseDouble(elements[j]);
                        }
                    } else if (fileName.equals("x.csv")) {
                        shape = reader.readLine().split(",");
                        nUsers = Integer.parseInt(shape[0]);
                        nFactor = Integer.parseInt(shape[1]);
                        x = new double[nUsers][nFactor];

                        for (int i = 0; i < x.length && (line = reader.readLine()) != null; ++i) {
                            String[] elements = line.split(",");
                            for (int j = 0; j < x[i].length; ++j)
                                x[i][j] = Double.parseDouble(elements[j]);
                        }
                    } else if (fileName.equals("y.csv")) {
                        shape = reader.readLine().split(",");
                        nItems = Integer.parseInt(shape[0]);
                        nFactor = Integer.parseInt(shape[1]);
                        y = new double[nItems][nFactor];

                        for (int i = 0; i < y.length && (line = reader.readLine()) != null; ++i) {
                            String[] elements = line.split(",");
                            for (int j = 0; j < y[i].length; ++j)
                                y[i][j] = Double.parseDouble(elements[j]);
                        }
                    } else if (fileName.equals("c.csv")) {
                        shape = reader.readLine().split(",");
                        nUsers = Integer.parseInt(shape[0]);
                        nItems = Integer.parseInt(shape[1]);
                        c = new double[nUsers][nItems];

                        for (int i = 0; i < c.length && (line = reader.readLine()) != null; ++i) {
                            String[] elements = line.split(",");
                            for (int j = 0; j < c[i].length; ++j)
                                c[i][j] = Double.parseDouble(elements[j]);
                        }
                    } else if (fileName.equals("p.csv")) {
                        shape = reader.readLine().split(",");
                        nUsers = Integer.parseInt(shape[0]);
                        nItems = Integer.parseInt(shape[1]);
                        p = new double[nUsers][nItems];

                        for (int i = 0; i < p.length && (line = reader.readLine()) != null; ++i) {
                            String[] elements = line.split(",");
                            for (int j = 0; j < p[i].length; ++j)
                                p[i][j] = Double.parseDouble(elements[j]);
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

    private void initWeights(double[][] matrix) {
        Random random = new Random();
        for (int i = 0; i < matrix.length; ++i)
            for (int j = 0; j < matrix[i].length; ++j)
                matrix[i][j] = Math.abs(random.nextDouble() * 0.01);
    }
}
