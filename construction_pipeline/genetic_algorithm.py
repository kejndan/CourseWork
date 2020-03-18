from deap import creator, base, tools
from sklearn.metrics import mean_squared_error
from inspect import types
import random
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from copy import deepcopy
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.pipeline import make_pipeline
from construction_pipeline.models import preprocessing_models, selection_models, classification_models, regression_models, clustering_models
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


CALLABLES = (types.FunctionType, types.MethodType)


def cv_silhouette_scorer(estimator, X) :
    # estimator.fit(X)
    return sklearn.metrics.silhouette_score(X, estimator.fit_predict(X))


def random_value_from(obj, name) :
    if isinstance(getattr(obj, name), np.ndarray) :
        return random.choice(getattr(obj, name))
    else :
        return getattr(obj, name)


class GeneticBase(object) :
    def __init__(self, type_explore, cv_func, score_func, n_generations, population_size, offspring_size, max_height,
                 min_height,
                 probability_mutation, probability_mate, path) :
        self.type_explore = type_explore
        self.n_generations = n_generations
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.max_height = max_height
        self.min_height = min_height
        self.probability_mutation = probability_mutation
        self.probability_mate = probability_mate
        self._control_population = []
        self._primitive_storage = None
        self._terminal_storage = None
        self.cv = 5
        self.time_table = []
        self.time_list = [0]
        self.cv_func = cv_func
        self.score_func = score_func
        self.path = path
        if self.type_explore == 'classification' :
            self.terminal_models = classification_models
        elif self.type_explore == 'regression' :
            self.terminal_models = regression_models
        else :
            self.terminal_models = clustering_models
            

    def _create_primitives_and_terminals_storage(self) :
        if self._primitive_storage is None :
            self._primitive_storage = []
            for name_transform, args_transform in preprocessing_models.items() :
                args_transform = deepcopy(args_transform)
                args_transform['type_transform'] = 'preprocessing'
                args_transform['name_transform'] = name_transform
                self._primitive_storage.append(type(name_transform, (), args_transform))
            for name_transform, args_transform in selection_models.items() :
                args_transform = deepcopy(args_transform)
                args_transform['type_transform'] = 'selection'
                args_transform['name_transform'] = name_transform
                self._primitive_storage.append(type(name_transform, (), args_transform))
        if self._terminal_storage is None :
            self._terminal_storage = []
            # if self.type_explore == 'classification':
            for name_transform, args_transform in self.terminal_models.items() :
                args_transform = deepcopy(args_transform)
                args_transform['type_transform'] = self.type_explore
                args_transform['name_transform'] = name_transform
                self._terminal_storage.append(type(name_transform, (), args_transform))

    def _setup_toolbox(self) :
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

    def _list_variables(self, obj) :
        return [i for i in dir(obj) if not i.startswith('__') and not isinstance(getattr(obj, i), CALLABLES)]

    def compare_inds(self, ind1, ind2) :
        if len(ind1) != len(ind2) :
            return False
        for i in range(len(ind1)) :
            if not isinstance(ind1[i][1], type(ind2[i][1])) :
                return False
        return True

    def add_info(self, individual) :
        names = []
        for model in individual :
            names.append(model[1].name_transform)
        individual.info = tuple(names)

    def _creation_expression(self) :
        expression_added = False
        while not expression_added :
            expression = []
            height = random.randint(self.min_height, self.max_height)
            for index_transforms in range(1, height) :
                primitive_obj = random.choice(self._primitive_storage)()
                for name_variables in self._list_variables(primitive_obj) :
                    if name_variables == 'estimator' :
                        # print(primitive_obj.estimator)
                        name_estimator = random.choice(list(primitive_obj.estimator.keys()))
                        args_estimator = primitive_obj.estimator[name_estimator]
                        args_estimator['name_transform'] = name_estimator
                        obj_transform = type(name_estimator, (), args_estimator)()
                        for name_variables_estimator in self._list_variables(obj_transform) :
                            setattr(obj_transform, name_variables_estimator,
                                    random_value_from(obj_transform, name_variables_estimator))
                        setattr(primitive_obj, name_variables, self._transform_to_sklearn(obj_transform))
                    else :
                        setattr(primitive_obj, name_variables, random_value_from(primitive_obj, name_variables))
                expression.append((index_transforms, primitive_obj))
            terminal_obj = random.choice(self._terminal_storage)()
            for name_variables in self._list_variables(terminal_obj) :
                setattr(terminal_obj, name_variables, random_value_from(terminal_obj, name_variables))
            expression.append((height, terminal_obj))
            count = 1
            for individual in self._control_population :
                if self.compare_inds(individual, expression) :
                    count += 1
            if count <= 0.01 * self.population_size or count <= 1 :
                expression_added = True
        return expression

    def _evaluation_individuals(self, population, pipeline_list, features, targets) :
        for number_pipeline, pipeline in enumerate(pipeline_list) :
            with warnings.catch_warnings() :
                # warnings.simplefilter('ignore')
                try :
                    avg_time = np.array(self.time_list).mean()
                    stop_time = avg_time if avg_time > 10 else 10
                    # start = time()
                    # @func_set_timeout(stop_time)
                    try :
                        if self.type_explore == 'clustering' :
                            # print(features)
                            # print(self.cv_func)
                            # cross_val_pipeline = cross_validate(pipeline[1],features, scoring=self.cv_func, cv=3)
                            cross_val_pipeline = func_timeout(stop_time, cross_validate,
                                                              kwargs={'estimator' : pipeline[1], 'X' : features,
                                                                      'scoring' : self.cv_func, 'cv' : self.cv})
                        else :
                            # print(features)
                            cross_val_pipeline = func_timeout(stop_time, cross_validate, args=(
                            pipeline[1], features, targets, None, self.cv_func, self.cv))

                        score = cross_val_pipeline['test_score'].mean()
                        time = cross_val_pipeline['score_time'].mean()
                        self.time_list.append(time)
                    # cross_val_pipeline = cross_validate(pipeline[1], features, targets, cv=self.cv, scoring='accuracy')
                    except FunctionTimedOut :
                        score = -float('inf')
                        time = float('inf')
                    # else:
                    #     score = cross_val_pipeline['test_score'].mean()
                    #     time = cross_val_pipeline['score_time'].mean()

                except Exception as e :
                    print(e)
                    score = -float('inf')
                    time = float('inf')
                finally :
                    # print(cross_val_pipeline['test_score'].mean(), type(cross_val_pipeline['test_score'].mean()))
                    # print( cross_val_pipeline['test_score'].mean() is np.float64('nan'))
                    print(number_pipeline, score)
                    population[number_pipeline].fitness.values = (len(population[number_pipeline]), score)
                    # print(population[number_pipeline].fitness.value)
                    population[number_pipeline].time = time

    def _population_to_sklearn(self, population) :
        pipeline_list = []
        for individual in population :
            pipeline = self.individual_to_sklearn(individual)
            pipeline_list.append((individual, pipeline))
        return pipeline_list

    def individual_to_sklearn(self, individual) :
        return make_pipeline(*[self._transform_to_sklearn(transform[1]) for transform in individual])

    def _transform_to_sklearn(self, transform, with_param=True) :
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
        common_transforms = []
        for transform_individual_1 in individual_1 :
            for transform_individual_2 in individual_2 :
                if isinstance(transform_individual_1[1], type(transform_individual_2[1])) :
                    common_transforms.append((transform_individual_1, transform_individual_2))
        transform = random.choice(common_transforms)
        if transform[0][1].type_transform != self.type_explore or not use_grid_search :
            param = random.choice(list(transform[0][1].__dict__.keys()))
            new_individual = deepcopy(individual_1)
            setattr(new_individual[transform[0][0] - 1][1], param, getattr(individual_2[transform[1][0] - 1][1], param))
            return new_individual
        param_transform_1 = transform[0][1].__dict__
        param_transform_2 = transform[1][1].__dict__
        param_grid = {}
        for name_param in param_transform_1.keys() :
            if param_transform_1[name_param] == param_transform_2[name_param] :
                param_grid[name_param] = [param_transform_1[name_param]]
            else :
                param_grid[name_param] = [param_transform_1[name_param], param_transform_2[name_param]]
        param_grid = deepcopy(param_grid)
        if len(param_grid) == 2 :
            return individual_1
        else :
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
        Данная функция выбираем из популяции двух случайных индивидов (благоприятных), у которых есть хотя бы одна одинковая трансформация
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
        if np.random.random() <= 1 / 3 :
            new_individual = self._replacement_mutation(individual)
        elif np.random.random() <= 2 / 3 :
            new_individual = self._shrink_mutation(individual)
        else :
            new_individual = self._insert_mutation(individual)
        self.add_info(new_individual)
        return new_individual

    def _replacement_mutation(self, individual) :
        """
        Данная функция заменяет случайную трансформацию на другую или меняем её параметр на другой, если она является
        примитивной, а если она терминальная, то меняем её аргумент на другой.
        :param individual: переданный индивид
        :return: новый выходной инвидид
        """
        # TODO сделать замену терминала на другой терминал
        # TODO ? сделать замену трансформации в случае если она не имеет параметров
        ind_copy = deepcopy(individual)  # копируем индивид
        random_transform = random.choice(ind_copy)  # выбираем из индивида случайную трансформацию
        if random_transform[0] < len(ind_copy) and np.random.random() <= 0.5 :
            # если трансформация является примитивом, то с вероятность 50%  мы заменяем эту трансформацию на другой
            # примитив

            # выбираем случайный примитив из набора классов и создаем объект
            primitive_obj = random.choice(self._primitive_storage)()

            # делаем так чтобы параметры примитива имели только одно значение
            for name_variables in self._list_variables(primitive_obj) :
                setattr(primitive_obj, name_variables, random_value_from(primitive_obj, name_variables))

            # заменяем нужную трансформацию на новый примитив
            ind_copy[random_transform[0] - 1] = (random_transform[0], primitive_obj)

        else :
            # если трансформация является терминалом или это примитив с вероятность 50%

            params = deepcopy(random_transform[1].__dict__)  # получаем параметры индивида

            # удаляем уточняющие параметры
            params.pop('type_transform')
            params.pop('name_transform')

            if params != {} :
                # если трансформация имеет параметры

                param = random.choice(list(params.keys()))  # выбираем случайный параметр
                if param != 'estimator' :
                    if random_transform[0] == len(ind_copy) :
                        # если трансформация является терминалом

                        # случайным образом выбираем новое значение для параметра из списка возможных
                        new_value = random.choice(self.terminal_models[random_transform[1].name_transform][param])

                    else :
                        # если трансформация является примитивом
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
        if len(ind_copy) > 1 :
            # если индивид имеет не только терминальную трансформацию

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
        # TODO вынести инициализацию нового в примитива в отдельную функцию

        ind_copy = deepcopy(individual)  # копируем индивид

        if len(ind_copy) < self.max_height :
            # если количество трансформаций в индивиде не превышает допустимый порог

            # выбираем случайный примитив из набора классов и создаем объект
            primitive_obj = random.choice(self._primitive_storage)()

            # делаем так чтобы параметры примитива имели только одно значение
            for name_variables in self._list_variables(primitive_obj) :
                setattr(primitive_obj, name_variables, random_value_from(primitive_obj, name_variables))

            ind_copy.insert(len(ind_copy) - 1, (len(ind_copy), primitive_obj))  # добавляем в индивид новый примитив
            ind_copy[-1] = (ind_copy[-1][0] + 1, ind_copy[-1][1])  # увеличиваем номер последний трансформации
        return ind_copy

    def _create_offspring(self, population, features, targets, time_info=True) :
        """
        Данная функция создает потомство.
        :param population: переданная популяция
        :return: потомство
        """
        # TODO перед скрещиванием проверить сразу на повторяемость
        population_copy = deepcopy(population)  # копируем популяцию
        offspring = []  # список для потоства
        if self.offspring_size is None :
            self.offspring_size = self.population_size
        time_table = []
        i = 0
        while i < self.offspring_size :
            begin_change = time()
            if np.random.random() <= self.probability_mate :
                type_change = 'mate'
                # c вероятностью self.probability_mate скрещиваем два индивида
                new_ind = self._toolbox.mate(*self._get_random_two_ind_for_mate(population), features, targets, False)
            elif np.random.random() <= self.probability_mate + self.probability_mutation :
                type_change = 'mutate'
                # c вероятностью self.probability_mutation мутируем индвид
                new_ind = self._toolbox.mutation(random.choice(population_copy))
            # считаем сколько раз подобный индивид повторяем в популяции и потомстве
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
                time_table.append([i, type_change, stop_change, stop_check])
                # если количество не превышает, то добавляем новый индивид в потомство
                offspring.append(new_ind)
                i += 1
        if time_info :
            for num in time_table :
                print(num)
        return offspring

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

    def fit(self, features, targets=[]) :

        # инициализация
        self._create_primitives_and_terminals_storage()
        self._setup_toolbox()
        self.population = self._toolbox.population()
        self.features_train = features
        self.targets_train = targets

        # добавляение информации об индивидах
        for ind in self.population :
            self.add_info(ind)

        with open(self.path+'\output.txt', 'w') as f:
            f.write('')
        pipeline_list_population = self._toolbox.compile(self.population)
        self._evaluation_individuals(self.population, pipeline_list_population, features, targets)
        for number_generation in range(self.n_generations) :
            with open(self.path + '\output.txt', 'a', encoding="UTF-8") as f:
                f.write('Поколение ' + str(number_generation) + ' ' +'!\n')
            # pipeline_list_population = self._toolbox.compile(self.population)
            # with open(self.path + '\output.txt', 'a', encoding="UTF-8") as f:
            #     f.write('Началось оценка популяции\n')
            # s = time()
            #
            with open(self.path + '\output.txt', 'a', encoding="UTF-8") as f:
                # f.write('Оценка окончена. Время оценки {0}\n'.format(time() - s))
                f.write('Началось создание потомков\n')
            s = time()
            offspring = self._create_offspring(self.population, features, targets, time_info=False)
            with open(self.path + '\output.txt', 'a', encoding="UTF-8") as f:
                f.write('Потомки созданы. Время создания {0}\n'.format(time() - s))
            pipeline_list_offspring = self._toolbox.compile(offspring)
            with open(self.path + '\output.txt', 'a', encoding="UTF-8") as f:
                f.write('Началось оценка популяции\n')
            s = time()
            self._evaluation_individuals(offspring, pipeline_list_offspring, features, targets)
            with open(self.path + '\output.txt', 'a', encoding="UTF-8") as f:
                f.write('Оценка окончена. Время оценки {0}\n'.format(time() - s))
            self.population[:] = self._toolbox.select(self.population + offspring, self.population_size)

        return self.population[0], self.individual_to_sklearn(self.population[0])

    def score(self, features_test, targets_test, time_work, output_inform) :
        pipeline_list_population = self._toolbox.compile(self.population)
        # print(pipeline_list_population)
        learned_pipelines = []
        # print(pipeline_list_population)

        x_valid, x_test, y_valid, y_test = train_test_split(features_test, targets_test, test_size=.5,
                                                            random_state=42)
        for index, pipeline in enumerate(pipeline_list_population) :
            try:
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
            except :
                pass
        learned_pipelines = sorted(learned_pipelines, key=lambda x : x[1], reverse=True)
        error_list = []
        for number in range(3) :
            if self.type_explore == 'regression':
                fitted_pipeline = pipeline_list_population[learned_pipelines[number][0]][1].fit(self.features_train,
                                                                                       self.targets_train)
                err = self.score_func(y_test, fitted_pipeline.predict(x_test)) ** (1 / 2)

            elif self.type_explore == 'clustering' :
                # fitted_pipeline = pipeline_list_population[learned_pipelines[0][0]][1].fit(self.features_train)
                err = sklearn.metrics.davies_bouldin_score(x_test, pipeline[1].fit_predict(x_test))
            else:
                fitted_pipeline = pipeline_list_population[learned_pipelines[number][0]][1].fit(self.features_train,
                                                                                           self.targets_train)
                err = self.score_func(y_test, fitted_pipeline.predict(x_test))
            error_list.append(err)

        print('Error', error_list)
        # output_inform[0].append(err)
        # output_inform[1].append(time_work)
        # print(learned_pipelines)
        f = open(self.path + '\\results.txt', 'w')
        for number in range(3):
            f.write('Pipeline {0}\n'.format(number + 1))
            for ind in self.population[learned_pipelines[number][0]]:

                f.write('{0} {1}\n'.format(ind[1].name_transform,ind[1].__dict__))
            f.write('Ошибка {0}\n'.format(error_list[number]))
                # f.write(str(ind[1].name_transform) + str(ind[1].__dict__) + )
        f.close()
        # self.export_information(learned_pipelines, err, time_work)
        for i, p in enumerate(self.population) :
            print(i, p, p.fitness.values)
        for i in learned_pipelines :
            print(i)
        with open(self.path + '\output.txt', 'a', encoding="UTF-8") as f :
            f.write('##END##')
    def export_information(self, learned_pipelines, error, time_work) :
        with open(self.name + str(datetime.today())[1 :] + '.txt', 'a') as f :
            for ind in self.population[learned_pipelines[0][0]] :
                f.write(str(ind[1].name_transform) + '\n')
                for key, val in ind[1].__dict__.items() :
                    f.write(key + ' ' + str(val) + '\n')
        print(self.name[:self.name.rfind('.')] + '_stats.txt')
        with open(self.name[:self.name.rfind('.')] + '_stats.txt', 'a') as f :
            f.write('MyAlg ' + str(error) + ' ' + str(time_work) + '\n')


class GeneticClassification(GeneticBase) :
    def __init__(self, n_generations=5, population_size=100, offspring_size=None, max_height=3, min_height=1,
                 probability_mutation=0.7, probability_mate=0.3, name=None, path=None) :
        self.type_explore = 'classification'
        self.cv_func = 'accuracy'
        self.score_func = accuracy_score
        self.name = name
        super().__init__(self.type_explore, self.cv_func, self.score_func, n_generations, population_size,
                         offspring_size, max_height, min_height,
                         probability_mutation, probability_mate, path)


class GeneticRegression(GeneticBase) :
    def __init__(self, n_generations=5, population_size=100, offspring_size=None, max_height=3, min_height=1,
                 probability_mutation=0.7, probability_mate=0.3, name=None, path=None) :
        self.type_explore = 'regression'
        self.cv_func = 'neg_mean_squared_error'
        self.score_func = mean_squared_error
        self.name = name
        super().__init__(self.type_explore, self.cv_func, self.score_func, n_generations, population_size,
                         offspring_size, max_height, min_height,
                         probability_mutation, probability_mate, path)


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
    name = 'Fish.csv'
    df = pd.read_csv('../datasets/'+name)
    # df = df.drop(df.index[150 :])
    df = df.drop(df.columns[0],1)
    x_train, x_test, y_train, y_test = train_test_split(df.drop(df.columns[-1], 1), df[df.columns[-1]], test_size=.2,
                                                        random_state=42)
    print(df)
    # GB = GeneticClustering(population_size=30, n_generations=5, name=name)
    # GB.cv = 3
    s = time()
    GB = GeneticRegression(population_size=30, n_generations=2, name=name, path='C:\\Users\\adels\PycharmProjects\project_coursework\\results')
    GB.cv = 3
    GB.fit(x_train,y_train)
    information_work = [[], []]
    print(x_train.shape, x_test.shape)
    GB.score(x_test,y_test,time()-s, information_work)
    print(time()-s)
    # print(p)
    #
    # y = GB.population
    # # x_train, x_test, y_train, y_test = train_test_split(df.drop(df.columns[-1], 1), df[df.columns[-1]], test_size=.2,
    # #                                                     random_state=42)
    # # print(y[0].fitness)
    # pipe = []
    # for em in y :
    #     pipe.append(GB.individual_to_sklearn(em))
    # l_p = []
    # for i, p in enumerate(pipe) :
    #     try :
    #         p.fit(x_train, y_train)
    #         l_p.append((i, p.score(x_test, y_test)))
    #     except :
    #         pass
    # k = sorted(l_p, key=lambda x : x[1], reverse=True)
    # for i, p in enumerate(y) :
    #     print(i, p, p.fitness.value)
    # for trans in y[k[0][0]] :
    #     print(trans[1], trans[1].__dict__)