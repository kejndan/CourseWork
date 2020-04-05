from deap import creator, base, tools
from sklearn.metrics import mean_squared_error
from inspect import types
import random
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from copy import deepcopy
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.pipeline import make_pipeline
from construction_pipeline.models import preprocessing_models, selection_models, classification_models,\
    regression_models, clustering_models
from time import time
import pandas as pd
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import cluster
from sklearn import naive_bayes
import sklearn
from sklearn import ensemble
import warnings
from sklearn.metrics import accuracy_score
from datetime import datetime
from func_timeout import func_timeout, FunctionTimedOut
from preprocessing_data import preprocessing
from tpot import TPOTClassifier, TPOTRegressor

CALLABLES = (types.FunctionType, types.MethodType)


def cv_silhouette_scorer(estimator, features) :
    return sklearn.metrics.silhouette_score(features, estimator.fit_predict(features))


def random_value_from(obj, name) :
    if isinstance(getattr(obj, name), np.ndarray) :
        return random.choice(getattr(obj, name))
    else :
        return getattr(obj, name)


class GeneticBase(object) :
    def __init__(self, type_explore, cv_func, score_func, n_generations, population_size, offspring_size, max_height,
                 min_height,
                 probability_mutation, probability_mate) :
        # параметры конструктора
        self.type_explore = type_explore
        self.n_generations = n_generations
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.max_height = max_height
        self.min_height = min_height
        self.probability_mutation = probability_mutation
        self.probability_mate = probability_mate
        if self.type_explore == 'classification' :
            self.terminal_models = classification_models
        elif self.type_explore == 'regression' :
            self.terminal_models = regression_models
        else :
            self.terminal_models = clustering_models
        self.cv = 5  # количество кросс-валидаций
        self.time_list = [0]  # список для хранения времени оценки каждого пайплайна
        self.cv_func = cv_func  # функция кросс-валидации
        self.score_func = score_func  # функция финальной оценки

        # служебные хранилища
        self.population = None

        self._control_population = []  # для контроля однотипных пайплайнов
        self._primitive_storage = None  # список классов примитивов
        self._terminal_storage = None  # список классов терминалов

    def _create_primitives_and_terminals_storage(self) :
        """
        Данная функция превращает словари в файле models.py в классы
        *primitive - это модели, которые не должны находится в конце пайплайна (обычно они выполняют
         какую-то предобработку данных)
        *terminals - это модели, которые могут находится в конце пайплайна, то есть обучающие модели
        **Создание классов происходит динамически
        """
        if self._primitive_storage is None :  # если список классов примитивов ещё не создан
            self._primitive_storage = []
            for name_transform, args_transform in preprocessing_models.items() :
                args_transform = deepcopy(args_transform)   # список возможных аргументов модели
                args_transform['type_transform'] = 'preprocessing'
                args_transform['name_transform'] = name_transform
                self._primitive_storage.append(type(name_transform, (), args_transform))
            for name_transform, args_transform in selection_models.items() :
                args_transform = deepcopy(args_transform)   # список возможных аргументов модели
                args_transform['type_transform'] = 'selection'
                args_transform['name_transform'] = name_transform
                self._primitive_storage.append(type(name_transform, (), args_transform))
        if self._terminal_storage is None :  # если список классов терминалов ещё не создан
            self._terminal_storage = []
            for name_transform, args_transform in self.terminal_models.items() :
                args_transform = deepcopy(args_transform)   # список возможных аргументов модели
                args_transform['type_transform'] = self.type_explore
                args_transform['name_transform'] = name_transform
                self._terminal_storage.append(type(name_transform, (), args_transform))

    def _setup_toolbox(self) :
        """
        Инициализация toolbox для использования алгоритмов библиотеки deap
        """
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
        creator.create('Individual', list, fitness=creator.FitnessMulti, info=list, time=int)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expression', self._creation_expression)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expression)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual, n=self.population_size)
        self._toolbox.register('evaluation', self._evaluation_individuals)
        self._toolbox.register('select', tools.selNSGA2)
        self._toolbox.register('mate', self._mate_operator)
        self._toolbox.register('mutation', self._mutate_operator)
        self._toolbox.register('compile', self._population_to_sklearn)

    def __list_variables(self, obj) :
        """
        Данная функция используется для получения переменных объекта
        :param obj: передаваемый объект
        :return: список переменных объекта
        """
        return [i for i in dir(obj) if not i.startswith('__') and not isinstance(getattr(obj, i), CALLABLES)]

    def _compare_inds(self, individual_1, individual_2) :
        """
        Данная функция проверяет индивиды совпадение всех трансформаций.
        :param individual_1: первый переданный индивид
        :param individual_2: второй переданный индивид
        :return: True - если индивиды совпадают, False - если нет.
        """
        if len(individual_1) != len(individual_2) :
            return False
        for i in range(len(individual_1)) :
            if not isinstance(individual_1[i][1], type(individual_2[i][1])) :
                return False
        return True

    def __add_info(self, individual) :
        """
        Данная функция используется для записи информации о пайплайне индивида
        :param individual: передаваемый индивид
        """
        names = []
        for model in individual :
            names.append(model[1].name_transform)
        individual.info = tuple(names)

    def __processing_hyperparameters_primitive(self, primitive_obj):
        """
        Данная функция инициализирует гиперпараметры из класса  в переданный объект
        :param primitive_obj: передаваемый объект
        """
        for name_variables in self.__list_variables(primitive_obj) :
            if name_variables == 'estimator' :
                name_estimator = random.choice(list(primitive_obj.estimator.keys()))
                args_estimator = primitive_obj.estimator[name_estimator]
                args_estimator['name_transform'] = name_estimator
                obj_transform = type(name_estimator, (), args_estimator)()
                for name_variables_estimator in self.__list_variables(obj_transform) :
                    setattr(obj_transform, name_variables_estimator,
                            random_value_from(obj_transform, name_variables_estimator))
                setattr(primitive_obj, name_variables, self._transform_to_sklearn(obj_transform))
            else :
                setattr(primitive_obj, name_variables, random_value_from(primitive_obj, name_variables))

    def _creation_expression(self) :
        """
        Данная функция создает пайплайн из примитивов и терминалов
        :return: пайплайн
        """
        expression_added = False
        while not expression_added :  # проверка на то был ли создан правильный пайплайн
            expression = []
            height = random.randint(self.min_height, self.max_height)  # выбираем случайную длину пайплайна h

            # на данном этапе мы заполняем пайплайн h -1 примитивом
            # данный цикл идёт от 1 до h - 1
            for index_transforms in range(1, height) :
                # выбираем случайный класс примитива после чего инициализируем объект и его гиперпараметры
                primitive_obj = random.choice(self._primitive_storage)()
                self.__processing_hyperparameters_primitive(primitive_obj)
                expression.append((index_transforms, primitive_obj))

            # заполняем пайплайн терминальной моделью
            # выбираем случайный класс терминала после чего инициализируем объект и его гиперпараметры
            terminal_obj = random.choice(self._terminal_storage)()
            for name_variables in self.__list_variables(terminal_obj) :
                setattr(terminal_obj, name_variables, random_value_from(terminal_obj, name_variables))
            expression.append((height, terminal_obj))

            # подсчёт сколько пайплайнов такого же типа уже имеется
            count = 1
            for individual in self._control_population :
                if self._compare_inds(individual, expression) :
                    count += 1
            # если количество одинковых пайплайнов меньше заданного процента или равно единице,
            # то добавляем созданный пайплайн
            if count <= 0.05 * self.population_size or count == 1 :
                expression_added = True
        self._control_population.append(expression)
        return expression

    def _evaluation_individuals(self, population, features, targets) :
        """
        Данная функция используется для оценки качества пайплайна (то есть его точности)
        :param population: переданная популяция
        :param features: X датасета
        :param targets: Y датасета
        :return:
        """
        pipeline_list = self._toolbox.compile(population)  # конвертация псевдо пайплайна в sklearn пайплайн
        for number_pipeline, pipeline in enumerate(pipeline_list) :
            with warnings.catch_warnings() :
                # warnings.simplefilter('ignore')  # скрытие предупреждений
                # подсчёт среднего времени кросс-валидации одного пайплайна
                avg_time = np.array(self.time_list).mean()*self.cv
                # установка максимального времени кросс-валидации одного пайплайна
                if len(self.time_list) > self.population_size*0.10 :
                    if avg_time < 1:
                        avg_time = 1
                    stop_time = avg_time * 10
                else:
                    stop_time = 600
                # попытка выполнить кросс-валидацию с помощью функции func_timeout, которая выбросит ошибку в случае
                # если время кросс-валидации превысит stop_time или если произойдет ошибка при исполнении, и в таком
                # score получается равный минус бесконечности, а время плюс бесконечности
                try :
                    if self.type_explore == 'clustering' :
                        # cross_val_pipeline = cross_validate(pipeline[1],features, scoring=self.cv_func, cv=3)
                        cross_val_pipeline = func_timeout(stop_time, cross_validate,
                                                          kwargs={'estimator' : pipeline[1], 'X' : features,
                                                                  'scoring' : self.cv_func, 'cv' : self.cv})
                    else :
                        cross_val_pipeline = func_timeout(stop_time, cross_validate, args=(
                        pipeline[1], features, targets, None, self.cv_func, self.cv))

                    score = cross_val_pipeline['test_score'].mean()
                    time = cross_val_pipeline['score_time'].mean()
                    self.time_list.append(time)
                except FunctionTimedOut :
                    score = -float('inf')
                    time = float('inf')
                except Exception as e :
                    # print(e)
                    score = -float('inf')
                    time = float('inf')
                finally :
                    print(number_pipeline, score, time)
                    population[number_pipeline].fitness.values = (len(population[number_pipeline]), score)
                    population[number_pipeline].time = time

    def _population_to_sklearn(self, population) :
        """
        Данная функция конвертирует псевдо пайплайны популяции в пайплайны sklearn
        :param population: переданная популяция из псевдо пайплайнов
        :return: популяция из sklearn пайплайнов
        """
        pipeline_list = []
        for individual in population :
            pipeline = self.individual_to_sklearn(individual)
            pipeline_list.append((individual, pipeline))
        return pipeline_list

    def individual_to_sklearn(self, individual) :
        """
        Данная функция конвертирует индивид, который является в псевдо пайплайном в пайплайн sklearn
        :param individual: переданный индивид
        :return: пайплайн sklearn
        """
        return make_pipeline(*[self._transform_to_sklearn(transform[1]) for transform in individual])

    def _transform_to_sklearn(self, transform, with_param=True) :
        """
        Данная функция превращает переданную модель в sklearn объект
        :param transform: переданная модель
        :param with_param: нужно ли удалять вспомогательные параметры
        :return: sklearn объект
        """
        if with_param :
            kwargs = deepcopy(transform.__dict__)
            if kwargs.get('type_transform') :
                kwargs.pop('type_transform')
            if kwargs.get('name_transform') :
                kwargs.pop('name_transform')
            return eval(transform.name_transform + '(**kwargs)')
        else :
            return eval(transform.name_transform + '()')

    def _mate_operator(self, individual_1, individual_2, features, targets, use_grid_search=False) :
        """
        Данная функция скрещевает два индивида
        :param individual_1: переданный первый индивид
        :param individual_2: переданный второй индивид
        :param features: X датасета
        :param targets: Y датасета
        :param use_grid_search: использовать grid_search для подбора оптимальных параметров при скрещивание
        :return: новый индивид порожденный скрщевинием ind_1 и ind_2
        """
        # поиск общих моделей в пайплайне
        common_transforms = []
        for transform_individual_1 in individual_1 :
            for transform_individual_2 in individual_2 :
                if isinstance(transform_individual_1[1], type(transform_individual_2[1])) :
                    common_transforms.append((transform_individual_1, transform_individual_2))

        transform = random.choice(common_transforms)  # случайный выбор общий модели
        # TODO if transform[0][1].type_transform != self.type_explore or not use_grid_search :
        # подготовка сетки параметров на тот случай если grid_search включен
        param_transform_1 = transform[0][1].__dict__
        param_transform_2 = transform[1][1].__dict__
        param_grid = {}
        for name_param in param_transform_1.keys() :
            if param_transform_1[name_param] == param_transform_2[name_param] :
                param_grid[name_param] = [param_transform_1[name_param]]
            else :
                param_grid[name_param] = [param_transform_1[name_param], param_transform_2[name_param]]
        param_grid = deepcopy(param_grid)

        # если grid_search выключен или параметров(кроме служебных) для настройки нет, иначе запускаем grid_search
        if not use_grid_search or len(param_grid) == 2:
            param = random.choice(list(transform[0][1].__dict__.keys()))
            new_individual = deepcopy(individual_1)
            setattr(new_individual[transform[0][0] - 1][1], param, getattr(individual_2[transform[1][0] - 1][1], param))
            return new_individual
        else:
            param_grid.pop('type_transform')
            param_grid.pop('name_transform')
            gscv = GridSearchCV(self._transform_to_sklearn(transform[0][1], False), param_grid, cv=3,
                                scoring='accuracy'). \
                fit(features, targets)
            new_params = gscv.best_params_
            new_params['type_transform'] = transform[0][1].type_transform
            new_params['name_transform'] = transform[0][1].name_transform
            new_individual = deepcopy(individual_1)
            new_individual[transform[0][0] - 1][1].__dict__ = new_params
            return new_individual

    def _get_random_two_ind_for_mate(self, population) :
        """
        Данная функция выбирает из популяции двух случайных индивидов (благоприятных),
         у которых есть хотя бы одна одинковая трансформация
        :param population: переданная популяция
        :return: кортеж из двух случайных индивидов
        """
        favorable_individuals = []  # список благоприятных индивидов

        # ищём благоприятные индивиды
        for i in range(len(population)) :
            for j in range(i + 1, len(population)) :
                # превращаем в множество список названий трансформаций
                a = set(population[i].info)
                b = set(population[j].info)

                if a.intersection(b) != set() :
                    # если есть общий трансформации, то добавляем пару индивидов
                    favorable_individuals.append((population[i], population[j]))
        random_inds = random.choice(favorable_individuals)  # случайным образом выбираем пару благоприятных индивидов
        return deepcopy(random_inds)

    def _mutate_operator(self, individual) :
        """
        Данная функция случайным образом определяет вид мутации индивида
        :param individual: переданный индивид
        :return: мутировавший индивид
        """
        if np.random.random() <= 1 / 3 :
            new_individual = self._replacement_mutation(individual)
        elif np.random.random() <= 2 / 3 :
            new_individual = self._shrink_mutation(individual)
        else :
            new_individual = self._insert_mutation(individual)
        self.__add_info(new_individual)
        return new_individual

    def _replacement_mutation(self, individual) :
        """
        Данная функция заменяет случайную модель на другую или меняет её параметр на другой,
         если она является примитивной, а если она терминальная, то меняет её аргумент на другой.
        :param individual: переданный индивид
        :return: новый выходной инвидид
        """
        # TODO ? сделать замену трансформации в случае если она не имеет параметров
        ind_copy = deepcopy(individual)  # копируем индивид
        random_transform = random.choice(ind_copy)  # выбираем из индивида случайную трансформацию
        if np.random.random() <= 0.5:
            if random_transform[0] < len(ind_copy):
                # если трансформация является примитивом, то с вероятность 50%  мы заменяем эту трансформацию на другой
                # примитив

                # выбираем случайный примитив из набора классов и создаем объект
                primitive_obj = random.choice(self._primitive_storage)()
                self.__processing_hyperparameters_primitive(primitive_obj)

                # заменяем нужную трансформацию на новый примитив
                ind_copy[random_transform[0] - 1] = (random_transform[0], primitive_obj)
            else:
                terminal_obj = random.choice(self._terminal_storage)()
                self.__processing_hyperparameters_primitive(terminal_obj)
                # заменяем нужную трансформацию на новый примитив
                ind_copy[random_transform[0] - 1] = (random_transform[0], terminal_obj)

        else :  # если трансформация является терминалом или это примитив с вероятность 50%
            params = deepcopy(random_transform[1].__dict__)  # получаем параметры индивида
            # удаляем уточняющие параметры
            params.pop('type_transform')
            params.pop('name_transform')

            if params != {} : # если трансформация имеет параметры
                param = random.choice(list(params.keys()))  # выбираем случайный параметр
                if param != 'estimator' :
                    if random_transform[0] == len(ind_copy) : # если трансформация является терминалом
                        # случайным образом выбираем новое значение для параметра из списка возможных
                        new_value = random.choice(self.terminal_models[random_transform[1].name_transform][param])
                    else : # если трансформация является примитивом
                        if random_transform[1].type_transform == 'selection' :
                            # случайным образом выбираем новое значение для параметра из списка возможных
                            new_value = random.choice(selection_models[random_transform[1].name_transform][param])
                        elif random_transform[1].type_transform == 'preprocessing' :
                            # случайным образом выбираем новое значение для параметра из списка возможных
                            new_value = random.choice(preprocessing_models[random_transform[1].name_transform][param])
                    setattr(random_transform[1], param, new_value)  # присваеваем новое значение параметру
        return ind_copy

    def _shrink_mutation(self, individual) :
        """
        Данная функция выбирает случайный примитив и удаляет его.
        :param individual: переданный примитив
        :return: новый выходной примитив
        """
        # TODO ? сделать замену на другой индивид если мутация не удалась
        ind_copy = deepcopy(individual)  # копируем индивид
        if len(ind_copy) > 1 : # если индивид имеет не только терминальную трансформацию
            del_transform = random.choice(ind_copy[:-1])  # случайным образом выбираем удаляемый примитив

            # меняем в индивиде номера трансформаций
            for number in range(del_transform[0], len(ind_copy)) :
                ind_copy[number] = (ind_copy[number][0] - 1, ind_copy[number][1])

            ind_copy.pop(del_transform[0] - 1)  # удаляем выбранный примитив
        return ind_copy

    def _insert_mutation(self, individual) :
        """
        Данная функция добавляет в индивид случайный примитив
        :param individual: переданный примитив
        :return: новый выходной примитив
        """
        # TODO ? сделать замену на другой индивид если мутация не удалась
        # TODO случайным образом вставлять новый примитив, а не в конец
        ind_copy = deepcopy(individual)  # копируем индивид

        if len(ind_copy) < self.max_height : # если количество трансформаций в индивиде не превышает допустимый порог

            # выбираем случайный примитив из набора классов и создаем объект
            primitive_obj = random.choice(self._primitive_storage)()
            # делаем так чтобы параметры примитива имели только одно значение
            self.__processing_hyperparameters_primitive(primitive_obj)

            ind_copy.insert(len(ind_copy) - 1, (len(ind_copy), primitive_obj))  # добавляем в индивид новый примитив
            ind_copy[-1] = (ind_copy[-1][0] + 1, ind_copy[-1][1])  # увеличиваем номер последний трансформации
        return ind_copy

    def _create_offspring(self, population, features, targets, time_info=False) :
        """
        Данная функция создает потомство на основании переданной популяции
        :param population: переданная популяция
        :param features: X датасета
        :param targets: Y датасета
        :return: потомство
        """
        # TODO перед скрещиванием проверить сразу на повторяемость
        population_copy = deepcopy(population)  # копируем популяцию
        offspring = []  # список для потоства
        if self.offspring_size is None :
            self.offspring_size = self.population_size
        time_table = []
        number = 0  # текущие количество потомков
        while number < self.offspring_size :
            begin_change = time()
            if np.random.random() <= self.probability_mate :
                # c вероятностью self.probability_mate скрещиваем два индивида
                type_change = 'mate'
                new_ind = self._toolbox.mate(*self._get_random_two_ind_for_mate(population), features, targets, False)
            elif np.random.random() <= self.probability_mate + self.probability_mutation :
                # c вероятностью self.probability_mutation мутируем индвид
                type_change = 'mutate'
                new_ind = self._toolbox.mutation(random.choice(population_copy))
            # считаем сколько раз подобный индивид повторяется в популяции и потомстве
            stop_change = time() - begin_change
            begin_check = time()
            count = 1
            for ind in offspring :
                if self._compare_inds(ind, new_ind) :
                    count += 1
            for ind in population_copy :
                if self._compare_inds(ind, new_ind) :
                    count += 1
            stop_check = time() - begin_check
            # проверка того чтобы количество новых потомков не превышало допустимую норму
            if count <= self.offspring_size * 0.1 or count <= 1 :
                time_table.append([number, type_change, stop_change, stop_check])
                # если количество не превышает, то добавляем новый индивид в потомство
                offspring.append(new_ind)
                number += 1
        if time_info :
            for num in time_table :
                print(num)
        return offspring

    def fit(self, features, targets=None) :
        """
        :param features:
        :param targets:
        :return:
        """
        if targets is None :
            targets = []
        if self.type_explore == 'regression':
            targets = np.array(targets, dtype=np.float)
        elif self.type_explore == 'classification':
            targets = np.array(targets, dtype=np.int)
        # инициализация
        self._create_primitives_and_terminals_storage()
        self._setup_toolbox()
        self.population = self._toolbox.population()
        self.features_train = features
        self.targets_train = targets

        # добавляение информации об индивидах
        for ind in self.population :
            self.__add_info(ind)
        print(f"Поколение 0 ",'!' * 100)

        print('Началось оценка популяции')
        s = time()
        self._evaluation_individuals(self.population, features, targets)
        print('Оценка окончена. Время оценки ', time() - s)
        for number_generation in range(self.n_generations) :
            if number_generation > 0:
                print(f'Поколение {number_generation} ', '!' * 100)
            print('Началось создание потомков')
            s = time()
            offspring = self._create_offspring(self.population, features, targets, time_info=False)
            print('Потомки созданы. Время создания ', time() - s)
            print('Началось оценка популяции')
            s = time()
            self._evaluation_individuals(offspring, features, targets)
            print('Оценка окончена. Время оценки ', time() - s)
            self.population[:] = self._toolbox.select(self.population + offspring, self.population_size)

        return self.population[0], self.individual_to_sklearn(self.population[0])

    def score(self, features_test, targets_test, time_work,output_inform, cross_val=False) :
        # смена типа данных для targets_test
        if self.type_explore == 'regression':
            targets_test = np.array(targets_test, dtype=np.float)
        elif self.type_explore == 'classification':
            targets_test = np.array(targets_test, dtype=np.int)

        # превращается популяцию из псевдо пайплайнов в sklearn пайплайны
        pipeline_list_population = self._toolbox.compile(self.population)
        learned_pipelines = []

        x_valid, x_test, y_valid, y_test = train_test_split(features_test, targets_test, test_size=.5)
        # валидация пайплайнов
        for index, pipeline in enumerate(pipeline_list_population) :
            try :
                if cross_val:
                    score_valid = cross_validate(pipeline[1],x_valid, y_valid,cv=5, scoring=(self.cv_func,))
                    ['test_'+self.cv_func].mean()
                    learned_pipelines.append((index, score_valid))
                else:
                    if self.type_explore == 'clustering' :
                        # pipeline[1].fit(self.features_train)
                        learned_pipelines.append(
                            (index, -sklearn.metrics.davies_bouldin_score(x_valid, pipeline[1].fit_predict(x_valid))))
                    elif self.type_explore == 'regression' :
                        pipeline[1].fit(self.features_train, self.targets_train)
                        learned_pipelines.append((index, -self.score_func(y_valid, pipeline[1].predict(x_valid))))
                    elif self.type_explore == 'classification' :
                        pipeline[1].fit(self.features_train, self.targets_train)
                        learned_pipelines.append((index, self.score_func(y_valid, pipeline[1].predict(x_valid))))
            except Exception as e:
                print(e)
                pass
        # проверка лучшего пайплайна на тестовой выборке
        learned_pipelines = sorted(learned_pipelines, key=lambda x : x[1], reverse=True)
        if self.type_explore == 'regression' :
            fitted_pipeline = pipeline_list_population[learned_pipelines[0][0]][1].fit(self.features_train,
                                                                                       self.targets_train)
            err = self.score_func(y_test, fitted_pipeline.predict(x_test)) ** (1 / 2)
        elif self.type_explore == 'clustering' :
            # fitted_pipeline = pipeline_list_population[learned_pipelines[0][0]][1].fit(self.features_train)
            err = sklearn.metrics.davies_bouldin_score(x_test, pipeline[1].fit_predict(x_test))
        else :
            fitted_pipeline = pipeline_list_population[learned_pipelines[0][0]][1].fit(self.features_train,
                                                                                       self.targets_train)
            err = self.score_func(y_test, fitted_pipeline.predict(x_test))

        print('Error', err)
        output_inform[0].append(err)
        output_inform[1].append(time_work)
        # print(learned_pipelines)
        for ind in self.population[learned_pipelines[0][0]] :
            print(ind[1].name_transform, ind[1].__dict__)

        # self.export_information(learned_pipelines, err, time_work)
        for i, p in enumerate(self.population) :
            print(i, p, p.fitness.values)
        for i in learned_pipelines :
            print(i)

    def export_information(self, learned_pipelines, error, time_work) :
        with open(self.name + '.txt', 'a') as f :
            for ind in self.population[learned_pipelines[0][0]] :
                f.write(str(ind[1].name_transform) + '\n')
                for key, val in ind[1].__dict__.items() :
                    f.write(key + ' ' + str(val) + '\n')
        print(self.name[:self.name.rfind('.')] + '_stats.txt')
        with open(self.name[:self.name.rfind('.')] + '_stats.txt', 'a') as f :
            f.write('MyAlg ' + str(error) + ' ' + str(time_work) + '\n')


class GeneticClassification(GeneticBase) :
    def __init__(self, n_generations=5, population_size=100, offspring_size=None, max_height=3, min_height=1,
                 probability_mutation=0.7, probability_mate=0.3, name=None) :
        self.type_explore = 'classification'
        self.cv_func = 'accuracy'
        self.score_func = accuracy_score
        self.name = name
        super().__init__(self.type_explore, self.cv_func, self.score_func, n_generations, population_size,
                         offspring_size, max_height, min_height,
                         probability_mutation, probability_mate)


class GeneticRegression(GeneticBase) :
    def __init__(self, n_generations=5, population_size=100, offspring_size=None, max_height=3, min_height=1,
                 probability_mutation=0.7, probability_mate=0.3, name=None) :
        self.type_explore = 'regression'
        self.cv_func = 'neg_mean_squared_error'
        self.score_func = mean_squared_error
        self.name = name
        super().__init__(self.type_explore, self.cv_func, self.score_func, n_generations, population_size,
                         offspring_size, max_height, min_height,
                         probability_mutation, probability_mate)


class GeneticClustering(GeneticBase) :
    def __init__(self, n_generations=5, population_size=100, offspring_size=None, max_height=3, min_height=1,
                 probability_mutation=0.7, probability_mate=0.3, name=None) :
        self.type_explore = 'clustering'
        self.cv_func = cv_silhouette_scorer
        self.score_func = sklearn.metrics.v_measure_score
        self.name = name
        super().__init__(self.type_explore, self.cv_func, self.score_func, n_generations, population_size,
                         offspring_size, max_height, min_height,
                         probability_mutation, probability_mate)


if __name__ == '__main__':
    name = 'movement_libras.data'

    information_work_sam = [[], []]
    information_work_tpot = [[], []]
    df = pd.read_csv('../datasets/'+name)
    # df = df.drop(df.index[1000 :])
    # df = df.drop(df.columns[-1],1)
    pp = preprocessing.PreProcessing(df, -1)
    pp.processing_missing_values()
    # pp.one_hot_encoder_categorical_features()
    df = pp.get_dataframe()
    print(df)
    for i in range(5):
        x_train, x_test, y_train, y_test = train_test_split(df.drop(df.columns[-1], 1), df[df.columns[-1]], test_size=.2)
        # GB = GeneticClustering(population_size=30, n_generations=5, name=name)
        # GB.cv = 3
        s = time()
        GB = GeneticClassification(population_size=30, n_generations=5, name=name)
        GB.cv = 3
        GB.fit(x_train, y_train)
        GB.score(x_test, y_test, time()-s, information_work_sam, cross_val=False)
        # print(time()-s)
        t1 = information_work_sam[1][-1]
        res1 = information_work_sam[0][-1]
        print(res1)
        print(t1)
        # print(np.array(information_work[0]).mean())
        # print(np.array(information_work[1]).mean())
        tpotr = TPOTClassifier(generations=5, population_size=30, verbosity=2, n_jobs=1)
        s = time()
        y_train = y_train.astype(np.int)
        tpotr.fit(x_train, y_train)
        t2 = time() - s
        res2 = tpotr.score(x_test, y_test)
        print(res2)
        print(t2)
        print('-'*100)
        information_work_tpot[0].append(res2)
        information_work_tpot[1].append(t2)
        with open('new_'+name[:name.rfind('.')] + '_stats.txt', 'a') as f :
            f.write('MyAlg ' + str(res1) + ' ' + str(t1) + '\n')
            f.write('TPOTAlg ' + str(res2) + ' ' + str(t2) + '\n')
    print(np.array(information_work_sam[0]).mean())
    print(np.array(information_work_sam[1]).mean())
    print(np.array(information_work_tpot[0]).mean())
    print(np.array(information_work_tpot[1]).mean())
    with open('new_' + name[:name.rfind('.')] + '_stats.txt', 'a') as f :
        f.write('\n')
        f.write('MyAlg ' + str(np.array(information_work_sam[0]).mean()) + ' ' + str(np.array(information_work_sam[1]).mean()) + '\n')
        f.write('TPOTAlg ' + str(np.array(information_work_tpot[0]).mean()) + ' ' + str(np.array(information_work_tpot[1]).mean()) + '\n')