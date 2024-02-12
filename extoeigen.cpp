#include "extoeigen.h"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

/* La primera función a realizar es la lectura del fichero .CSV (dataset):
   es un vector de vectores de tipo string.
   Leerá línea por línea y almacenará en un vector de vectores del tipo string */
std::vector<std::vector<std::string>> ExToEigen::leerCSV() {
    /* Abrir archivo para lectura solamente */
    std::ifstream Fichero(setDatos);
    /* Vector de vectores del tipo string: tendrá los datos del dataset */
    std::vector<std::vector<std::string>> datosString;
    /* Se itera a través de cada línea del dataset, y se divide el contenido
        usando el delimitador provisto por el constructor */
    std::string linea = "";
    while(getline(Fichero, linea)) {
        std::vector<std::string> vectorFila;
        boost::algorithm::split(vectorFila, linea, boost::is_any_of(colDelimitador));
        datosString.push_back(vectorFila);
    }
    /* Se cierra el fichero */
    Fichero.close();
    /* Se retorna el vector de vectores */
    return datosString;
}
/* Segunda función para guardar el vector de vectores del tipo String.
   La idea es presentarlos como un Dataframe (los datos) */
Eigen::MatrixXd ExToEigen::CSVtoEigen(std::vector<std::vector<std::string>> setDatos, int rows, int cols) {
    /* Si se tiene cabecera se remueve, se manipula solo los datos */
    if (header==true) {
        rows -= 1;
    }
    /* Se itera sobre filas y columnas para almacenar en la matriz vacía de tamaño filas por columnas.
       Básicamente se alacenará string en el vector, luego se pasan a flotantes (float) para ser manipulados */
    Eigen::MatrixXd matrizDataFrame(cols, rows);
    int i, j;
    for (i=0; i<rows; i++) {
        for (j=0; j<cols; j++) {
            matrizDataFrame(j, i) = atof(setDatos[i][j].c_str());  // Se guardan del tipo float (atof)
        }
    }
    /* Se transpone la matriz para tener filas por columnas*/
    return matrizDataFrame.transpose();
}

/* Para desarrollar el algoritmo de machine learning, el cual será regresión lineal por mínimos cuadrados
   ordinarios, se usarán los datos del dataset (winedata.csv) el cual se realizará para múltiples variables.
   Dada la naturaleza de RL, si se tiene valores con diferentes unidades (órdenes de magnitus), una variable
   podría beneficiar/estropear otra(s) variable(s): Se necesitará estandarizar los datos, dando a todas las
   variables el mismo orden de magnitud y centradas en 0. Para ello construiremos una función de normalización
   basada en el Z-Score. Se necesitan entonces 3 funciones: promedio, desviación estándar y la normalización
   Z-Score. */

/* La palabra clave auto especifica que el tipo de la variable que se empueza a declarar se deducirá
   automáticamente de su inicializador y, para las funciones, si su tipo de retorno es auto, se evaluará
   mediante la expresión del tipo de retorno en tiempo de ejecución. */

/* auto ExToEigen::Promedio(Eigen::MatrixXd datos) {
    return datos.colwise().mean();
} */

/* En C++ la herencia del tipo de dato no es directa o no se dabe que tipo de dato debe retornar
   entonces se declara el tipo en una expresión "decltype" con el fin de tener seguridad de que
   tipo de dato retornará la función. */
auto ExToEigen::Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean()) {
    return datos.colwise().mean();
}

/* Para implementar la función de desviación estandar los datos serán Xi - Promedio y de esa forma obtener
   la desviación estándar. */
auto ExToEigen::DesviacionEstandar(Eigen::MatrixXd datos) -> decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt()){
    return ((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt();
}

/* Acto seguido se necesita aplicar el promedio de la desviación estándar para hacer la normalización Z-Score. */
Eigen::MatrixXd ExToEigen::Normalizacion(Eigen::MatrixXd datos) {
    //auto promedio = Promedio(datos);
    /* Hacemos la diferencia Xi - Promedio */
    Eigen::MatrixXd diferenciaPromedio = datos.rowwise()-Promedio(datos);
    /* Se aplica la desviación estándar */
    //auto desviacionEst = DesviacionEstandar(diferenciaPromedio);
    /* Se hace la normalización de la matriz de desviación de datos*/
    Eigen::MatrixXd matrixNorm = diferenciaPromedio.array().rowwise()/DesviacionEstandar(diferenciaPromedio).array();
    return matrixNorm;
}

/* Se implementa la función para dividir el conjunto de datos en entrenamiento y prueba.
   En el Dataset se observan 12 columnas o variables. Las 11 primeras columnas corresponden a las variables independientes
   identificadas en la literatura como "FEATURES". La última columna (12va columna) corresponde a la variable dependiente
   conocida en la literatura como "TARGET".
   La función debe retornar 4 conjuntos de datos, a saber:
   X_train: conjunto de datos de entrenamiento de las features.
   y_train: conjunto de datos de entrenamiento de la target.
   X_test: conjunto de datos de prueba de las features.
   y_test: conjunto de datos de prueba de la target. */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ExToEigen::TrainTestSplit(Eigen::MatrixXd datos, float sizeTrain) {
    int filas = datos.rows();
    int filasEntrenamiento = round(filas*sizeTrain);
    int filasPrueba = filas-filasEntrenamiento;
    /* Con Eigen se puede especificar un bloque de una matriz: por ejemplo, se puede seleccionar las filas superiores
       para el conjunto de entrenamiento indicando cuantas filas se desea, seleccionando desde 0. */
    Eigen::MatrixXd entrenamiento = datos.topRows(filasEntrenamiento);
    /* Una vez seleccionada las filas superiores, se seleccionan las columnas de la izquierda,
       correspondientes a las features o variables independientes. */
    Eigen::MatrixXd X_train = entrenamiento.leftCols(datos.cols()-1);
    /* Seleccionamos la variable dependiente o target, que corresponde a la última columna */
    Eigen::MatrixXd y_train = entrenamiento.rightCols(1);
    /* Seguidamente se repite el procedimiento para el conjunto de pruebas */
    Eigen::MatrixXd pruebas = datos.bottomRows(filasPrueba);
    Eigen::MatrixXd X_test = pruebas.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_test = pruebas.rightCols(1);
    /* Al retornar la tupla se empaqueta (make_tuple) para enviarse como objeto */
    return std::make_tuple(X_train, y_train, X_test, y_test);
}

/* Se crean dos funciones para exportar los valores*/

void ExToEigen::VectorToFile(std::vector<float> vector, std::string nombreFile){
    std::ofstream ficheroSalida(nombreFile);
    std::ostream_iterator<float> iteradorSalida(ficheroSalida, "\n");
    std::copy(vector.begin(), vector.end(), iteradorSalida);
}

void ExToEigen::EigenToFile(Eigen::MatrixXd datos, std::string nombreFile){
    std::ofstream ficheroSalida(nombreFile);
    if(ficheroSalida.is_open()){
        ficheroSalida << datos << "\n";
    }
}

















