import random
import math
from sklearn.model_selection import cross_val_score


def objective(model, params, X, y, cv=3, scoring='f1', n_jobs=-1):
    """
    Calcula la puntuación media obtenida mediante validación cruzada
    para un modelo de clasificación con parámetros específicos.

    Parámetros:
    model: Clase del modelo de clasificación a utilizar.
    params: Parámetros del modelo.
    X: Datos de entrada.
    y: Etiquetas correspondientes a los datos de entrada.
    cv: Estrategia de validación cruzada.
    scoring: Métrica de evaluación a utilizar.
    n_jobs: Número de trabajos a ejecutar en paralelo durante la validación cruzada.

    Retorna:
    float
        Puntuación media obtenida mediante validación cruzada para el modelo dado.
    """
    # Creamos una instancia del modelo de clasificación con los parámetros dados
    classifier_knn = model(**params)

    # Realizamos la validación cruzada
    score = cross_val_score(classifier_knn, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)

    # Devolvemos la puntuación media
    return score.mean()


def generate_initial_population(dictionary_of_values, number_population=100):
    """
    Genera una población inicial de individuos aleatorios basados en un diccionario
    de valores posibles para cada característica.

    Parámetros:
    dictionary_of_values: Diccionario que contiene para cada característica (clave) una lista de posibles valores
        (valores).
    number_population: Número de individuos a generar en la población inicial.

    Retorna:
    list
        Lista de diccionarios, donde cada diccionario representa un individuo de la población.
    """
    population = []
    keys = list(dictionary_of_values.keys())

    # Itera sobre el número de individuos de la población inicial
    for i in range(number_population):
        new_dict = {}
        # Para cada característica, elige aleatoriamente uno de los valores posibles
        for key in keys:
            new_dict[key] = random.choice(dictionary_of_values[key])
        # Agrega el nuevo individuo a la población
        population.append(new_dict)

    return population


def reproduction(dict1, dict2):
    """
    Realiza una operación de reproducción entre dos individuos, representados
    como diccionarios, para generar dos nuevos individuos.

    Parámetros:
    dict1: Primer individuo (diccionario).
    dict2: Segundo individuo (diccionario).

    Retorna:
    tuple
        Tupla que contiene dos nuevos individuos (diccionarios).
    """

    keys = list(dict1.keys())

    # Mezcla las claves para seleccionar aleatoriamente de cada diccionario
    random.shuffle(keys)

    # Crea dos nuevos diccionarios para los individuos generados
    mixed_dict1 = {}
    mixed_dict2 = {}

    # Itera sobre las claves mezcladas
    for key in keys:
        # Selecciona aleatoriamente de qué diccionario tomar el valor
        if random.choice([True, False]):
            mixed_dict1[key] = dict1[key]
            mixed_dict2[key] = dict2[key]
        else:
            mixed_dict1[key] = dict2[key]
            mixed_dict2[key] = dict1[key]

    return mixed_dict1, mixed_dict2


def mutate_chromosome_with_temperature(chromosome, temperature, dictionary_of_values):
    """
    Realiza una mutación en un cromosoma (representado como un diccionario) con una probabilidad
    controlada por una temperatura. La mutación consiste en cambiar uno de los valores del cromosoma
    por otro valor posible según un diccionario de valores.

    Parámetros:
    chromosome: Cromosoma a mutar, representado como un diccionario.
    temperature: Temperatura que controla la probabilidad de aceptar la mutación.
    dictionary_of_values: Diccionario que contiene para cada característica (clave) una lista de
        posibles valores (valores).

    Retorna:
    dict
        El cromosoma mutado (puede ser igual al original si la mutación no se realiza).
    """

    out_mutation = chromosome.copy()
    # Calcula la probabilidad de aceptar la mutación utilizando la temperatura
    mutation_probability = math.exp(-1 * (1 / temperature))

    # Si la probabilidad de mutación es mayor que un valor aleatorio, realiza la mutación
    if random.random() < mutation_probability:
        keys = list(out_mutation.keys())
        mutation_key = random.choice(keys)
        mutation = random.choice(dictionary_of_values[mutation_key])

        # Realiza la mutación en el parámetro elegido
        out_mutation[mutation_key] = mutation

    return out_mutation


def genetic_hyper(model, X, y, dictionary_of_values, cv=5, scoring='f1',
                  max_iterations=100, number_initial_population=100, initial_temperature=1.0,
                  cooling_rate=0.95, n_jobs=-1):
    """
    Realiza una búsqueda de hiperparámetros utilizando un algoritmo genético.

    Parámetros:
    model: Clase del modelo de clasificación a utilizar.
    X: Datos de entrada.
    y: Etiquetas correspondientes a los datos de entrada.
    dictionary_of_values: Diccionario que contiene para cada hiperparámetro (clave) una lista de
        posibles valores (valores).
    cv: Estrategia de validación cruzada.
    scoring: Métrica de evaluación a utilizar.
    max_iterations: Número máximo de iteraciones del algoritmo genético.
    number_initial_population: Tamaño de la población inicial.
    initial_temperature: Temperatura inicial para la mutación de los cromosomas.
    cooling_rate: Tasa de enfriamiento para ajustar la temperatura en cada iteración.
    n_jobs: Número de trabajos a ejecutar en paralelo durante la validación cruzada.

    Retorna:
    tuple
        Tupla que contiene los mejores parámetros encontrados y su puntuación correspondiente.
    """

    # Inicializamos
    current_population = generate_initial_population(dictionary_of_values, number_population=number_initial_population)

    best_cost = 0
    temperature = initial_temperature
    best_params = current_population[0]

    number_population = len(current_population)

    # Iteramos hasta max_iterations
    for iteration in range(max_iterations):

        # Para cada individuo de la población, aplicamos validación cruzada
        actual_cost_list = [objective(model, params, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs) for params in
                            current_population]

        for index, cost in enumerate(actual_cost_list):
            if cost > best_cost:
                best_params = current_population[index]
                best_cost = cost
                print(f"En generación {iteration}, la mejor metrica {scoring} es {best_cost}, y los parametros son "
                      f"{best_params}")

        # Ordenamos los individuos en función de la puntuación (de mayor a menor)
        index_list = sorted(range(len(actual_cost_list)), key=lambda x: actual_cost_list[x], reverse=True)

        current_population = [current_population[k] for k in index_list]
        current_population = current_population[:number_population]

        # Reproducimos a los alpha (los mejores) con varios de la población
        sibling_chromosomes_list = []
        for i in range(5):
            alpha = current_population[i]
            for j in range(6 - i):
                selected = random.choice(current_population[i:])

                # Los reproducimos
                chromosome_1, chromosome_2 = reproduction(alpha, selected)

                # Mutamos algunos cromosomas utilizando la temperatura
                chromosome_1 = mutate_chromosome_with_temperature(chromosome_1, temperature, dictionary_of_values)
                chromosome_2 = mutate_chromosome_with_temperature(chromosome_2, temperature, dictionary_of_values)

                # Y los incorporamos en la nueva generación
                sibling_chromosomes_list.extend([chromosome_1, chromosome_2])

        # Reproducimos en orden de los mejores primero y de ahi para abajo
        for index_a in range(0, len(current_population), 2):
            # Obtenemos los cromosomas de cada estado
            chromosome_a = current_population[index_a]
            chromosome_b = current_population[index_a + 1]

            # Y los reproducimos
            chromosome_1, chromosome_2 = reproduction(chromosome_a, chromosome_b)

            # Mutamos algunos cromosomas utilizando la temperatura
            chromosome_1 = mutate_chromosome_with_temperature(chromosome_1, temperature, dictionary_of_values)
            chromosome_2 = mutate_chromosome_with_temperature(chromosome_2, temperature, dictionary_of_values)

            # Y los incorporamos en la nueva generación
            sibling_chromosomes_list.extend([chromosome_1, chromosome_2])

        # Creamos la nueva generación de estados
        current_population = sibling_chromosomes_list[:number_population]

        # Enfriamos la temperatura
        temperature *= cooling_rate

    # Si alcanzamos el número máximo de iteraciones, devolvemos los mejores parámetros encontrados
    return best_params, best_cost
