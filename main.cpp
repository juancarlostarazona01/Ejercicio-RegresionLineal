#include "extoeigen.h"
#include "modellinealregression.h"

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

/*  Se requiere crear una aplicacion que lea ficheros que contengan set de datos
 *  (dataSet) en CSV debe presentar una clase que represente la extraccion de los datos
 *  entre la carga de los datos, la normalizacion de los datos y la manipulacion
 *  de los datos con Eigen: ExToEigen
 *  Adicional se requiere otra clase que presente el calculo de la regresion lineal*/

int main(int argc, char *argv[])
{
    /* Se crea un objeto de tipo ExToEigen, en el cual se incluyen los 3 argumentos
     * requeridos por el constructor (NombreDataset, Delimitador y el Header) */
    ExToEigen extraction(argv[1], argv[2], argv[3]);
    ModelLinealRegression LR;
    /* A continuación, se leen los datos del fichero por la función leerCSV() */
    std::vector<std::vector<std::string>> data = extraction.leerCSV();
    /* Para probar la segunda función se define la cantidad de filas y columnas basados en los datos de entrada */
    int rows = data.size()+1;
    int cols = data[0].size();

    Eigen::MatrixXd matrizDF = extraction.CSVtoEigen(data, rows, cols);
    //auto promedio = extraction.Promedio(matrizDF);
    Eigen::MatrixXd diferenciaPromedio = matrizDF.rowwise()-extraction.Promedio(matrizDF);
    //auto desviacionEst = extraction.DesviacionEstandar(diferenciaPromedio);

    Eigen::MatrixXd matrizNorm = extraction.Normalizacion(matrizDF);

    /* Se requiere verificar la función TrainTestSplit si la división del número de filas y columnas
     * es el esperado para todos los conjuntos de datos (X_train, y_train, X_test, y_test) */
    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> divDatos = extraction.TrainTestSplit(matrizNorm, 0.8);
    /* Se desempaca la tupla del objeto: https://www.cplusplus.com/reference/tuple/tuple/ */
    std::tie(X_train, y_train, X_test, y_test) = divDatos;
    /* Inspección visual de columnas y filas en entrenamiento */
    /*std::cout << "Total de filas:                   " << matrizNorm.rows() << std::endl;
    std::cout << "Total de columnas:                " << matrizNorm.cols() << std::endl;
    std::cout << "Total filas entrenamiento F:      " << X_train.rows() << std::endl;
    std::cout << "Total columnas entrenamiento F:   " << X_train.cols() << std::endl;
    std::cout << "Total filas entrenamiento T:      " << y_train.rows() << std::endl;
    std::cout << "Total columnas entrenamiento T:   " << y_train.cols() << std::endl;
    std::cout << "========================================================================================================================" << std::endl;
    std::cout << "Total de filas:                   " << matrizNorm.rows() << std::endl;
    std::cout << "Total de columnas:                " << matrizNorm.cols() << std::endl;
    std::cout << "Total filas prueba F:             " << X_test.rows() << std::endl;
    std::cout << "Total columnas prueba F:          " << X_test.cols() << std::endl;
    std::cout << "Total filas prueba T:             " << y_test.rows() << std::endl;
    std::cout << "Total columnas prueba T:          " << y_test.cols() << std::endl;*/

    /*std::cout << "Promedio: \n" << extraction.Promedio(matrizDF) << std::endl;
    std::cout << "========================================================================================================================" << std::endl;
    std::cout << "Desviación estándar: \n" << extraction.DesviacionEstandar(diferenciaPromedio) << std::endl;*/

    /* Se crean dos vectores para prueba y entrenamiento en unos para probar el modelo
    * de regresión lineal*/
    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());

    /* Redimensión de la matriz para ubicarlas dentro del vector */
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vectorTrain;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vectorTest;

    /* Se define el vector theta inicial como vector de ceros para pasarlo a la función
    * del gradiente descendiente*/
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    float alpha = 0.01;
    int iteraciones = 1000;

    /* Se definen las variables de salida que representan los coeficientes y el vector
    * de la función de costo */
    Eigen::VectorXd thetaOut;
    std::vector<float> costo;

    std::tuple<Eigen::VectorXd, std::vector<float>> gradienteD = LR.GradienteDescendiente(X_train, y_train, theta, alpha, iteraciones);
    std::tie(thetaOut, costo) = gradienteD;

    /* Se imprimen los coeficientes*/
    /*std::cout <<thetaOut<< std::endl;
    for(auto valor: costo){
        std::cout<<valor<<std::endl;
    }*/

    /*Se exportan los valores a ficheros*/

    //extraction.VectorToFile(costo,"costo.txt");
    //extraction.EigenToFile(thetaOut,"thetaout.txt");

    /*Se calcula a base de predicciones el entrenamiento*/
    auto mu_data = extraction.Promedio(matrizDF);
    auto mu_features = mu_data(0,11);
    auto escalado = matrizDF.rowwise()-matrizDF.colwise().mean();
    auto sigmaData = extraction.DesviacionEstandar(escalado);
    auto sigmaFeatures = sigmaData(0,11);
    Eigen::MatrixXd y_Train_Hat = (X_train*thetaOut*sigmaFeatures).array()+mu_features;
    Eigen::MatrixXd y = matrizDF.col(11).topRows(1279);

    /*Se crea la variable que representa r2*/

    float R2 = LR.R2Cuadrado(y, y_Train_Hat);

    std::cout << R2 <<std::endl;

    extraction.EigenToFile(y_Train_Hat,"y_Train_hat.txt");
}



