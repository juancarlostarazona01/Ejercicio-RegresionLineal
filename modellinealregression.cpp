#include "modellinealregression.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

/* Se necesita entrenar el modelo, lo que implica minimizar la función de costo
 * de esta forma se puede medir la función de hipotesis.
   la función de costo es la forma de penalizar al modelo por cometer un error*/

float ModelLinealRegression::FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta){
    Eigen::MatrixXd diferencia = pow((X*theta-y).array(), 2);
    return (diferencia.sum()/(2*X.rows()));
}

/* se necesita proveer al algoritmo una función para dar los valores iniciales de theta,
 * el cual variará o cambiará iterativamente hasta que converja al valor minimo de
 * nuestra función de costo. Basicamente esto representa el gradiente descendiente, el cual
 * es las derivadas parciales de la función. Las entradas para la función seran X (features),
 * y (target), alpha (learning rate) y el numero de iteraciones (numero de veces que se actulizara
 * theta hasta que la función converja.*/

std::tuple<Eigen::VectorXd, std::vector<float>> ModelLinealRegression::GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iteraciones){
    /* Almacenamiento temporal de thetas */
    Eigen::MatrixXd temporal = theta;
    /* Necesitamos la cantidad de parametros m (features)*/
    int parametros = theta.rows();
    /* Costo inicial: Se actualizara con los nuevos pesos */
    std::vector<float> costo;
    costo.push_back(FuncionCosto(X, y, theta));
    /* Por cada iteración se calcula la función de error. Se actualiza theta y se calcula
       el nuevo valor de la función de costo para los nuevos valores de theta*/
    for(int i=0; i<iteraciones; ++i){
        Eigen::MatrixXd error = X*theta - y;
        for(int j=0; j<parametros; ++j){
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j, 0) = theta(j, 0) - ((alpha/X.rows())*termino.sum());
        }
        theta = temporal;
        costo.push_back(FuncionCosto(X, y, theta));
    }
    return std::make_tuple(theta, costo);

}

/*Se crea la metrica r2
Si*/
float ModelLinealRegression::R2Cuadrado(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
    auto numerador = pow((y-y_hat).array(),2).sum();
    auto denominador = pow(y.array()-y.mean(),2).sum();
    return 1-(numerador/denominador);
}
