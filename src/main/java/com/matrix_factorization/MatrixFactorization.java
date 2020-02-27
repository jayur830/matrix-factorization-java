package com.matrix_factorization;

public interface MatrixFactorization {
    void setPrintLog(boolean printLog);
    void fit(int stepSize);
    double[][] predict();
    double[][] getR();
    void put(int user, int item, double rating);
    void addUser(double[] user);
    void addUsers(double[][] _users);
    void addEmptyUser();
    void addEmptyUsers(int nUsers);
    void addEmptyItem();
    void addEmptyItems(int nItems);
    void saveModel(String fileName);
    void loadModel(String modelPath);
}
