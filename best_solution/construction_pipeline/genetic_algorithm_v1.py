from deap import creator, base, tools, gp
import random
import numpy as np
import pandas as pd
from construction_pipeline.models import preprocessing_models, selection_models, classification_models
from copy import deepcopy,copy
# from sklearn import preprocessing
# from sklearn import feature_selection
import sklearn
# from sklearn import ensemble
from sklearn.pipeline import  make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from inspect import types
from sklearn.model_selection import GridSearchCV
from time import time

def random_value_from(obj, name):
    if isinstance(getattr(obj, name), np.ndarray):
        return random.choice(getattr(obj, name))
    else:
        return getattr(obj, name)

class GeneticBase(object):
    def __init__(self, generations=100, population_size=100, offspring_size=None):
        self.generations = generations
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.base_primitives = None
        self.max_height = 3
        self.min_height = 1
        self.probability_mutation = 0.7
        self.probability_mate = 0.3
        self.__population = []
        self.control_pop = []

    def create_base_primitives_and_terminals(self):
        if self.base_primitives is None:
            self._base_primitives = []
            for model_name, model_args in preprocessing_models.items():
                model_args = deepcopy(model_args)
                # model_args['random_value_from'] = random_value_from
                model_args['type_model'] = 'preprocessing'
                model_args['name_model'] = model_name
                self._base_primitives.append(type(model_name, (), model_args))
            for model_name, model_args in selection_models.items() :
                model_args = deepcopy(model_args)
                # model_args['random_value_from'] = random_value_from
                model_args['type_model'] = 'selection'
                model_args['name_model'] = model_name
                self._base_primitives.append(type(model_name, (), model_args))
            self._base_terminals = []
            for model_name, model_args in classification_models.items() :
                model_args = deepcopy(model_args)
                # model_args['random_value_from'] = random_value_from
                model_args['type_model'] = 'classification'
                model_args['name_model'] = model_name
                self._base_terminals.append(type(model_name, (), model_args))





    def _setup_toolbox(self):
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
        creator.create('Individual', list, fitness=creator.FitnessMulti, info=tuple)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expression', self._grow_tree)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expression)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual, n = self.population_size)
        self._toolbox.register('evaluation', self._evaluation_individuals)
        self._toolbox.register('select', tools.selNSGA2)
        self._toolbox.register('mate', self._mate_operator)
        self._toolbox.register('mutate', self._mutate_operator)
        # self._toolbox.register('compilation', self.expression_to_sklearn)


    def _grow_tree(self):
        add = False
        CALLABLES = (types.FunctionType, types.MethodType)
        while not add:
            expression = []
            height = random.randint(self.min_height, self.max_height)
            for i in range(1,height):
                primitive_obj = random.choice(self._base_primitives)()
                for key in [i for i in dir(primitive_obj) if not i.startswith('__') and not isinstance(getattr(primitive_obj,i), CALLABLES)]:
                    setattr(primitive_obj, key, random_value_from(primitive_obj, key))
                expression.append((i, primitive_obj))
            terminal_obj = random.choice(self._base_terminals)()
            for key in [i for i in dir(terminal_obj) if not i.startswith('__') and not isinstance(getattr(terminal_obj,i), CALLABLES)]:
                t = random_value_from(terminal_obj, key)
                setattr(terminal_obj, key, t)
            expression.append((height, terminal_obj))
            count = 0
            for ind in self.control_pop:
                if self.compare_inds(ind,expression):
                    count += 1
            if count <= 3:
                add = True
        self.control_pop.append(expression)
        return expression

    def model_to_sklearn(self, model, with_param=True):
        if with_param:
            m = model
            t = model.__dict__
            kwargs = copy(t)
            kwargs.pop('type_model')
            kwargs.pop('name_model')
            return eval(model.name_model+'(**kwargs)')
        else:
            return eval(model.name_model+'()')

    def individual_to_pipeline(self, ind):
        pipeline = make_pipeline(*[self.model_to_sklearn(model[1]) for model in ind])
        return pipeline

    def _evaluation_individuals(self, population, pipeline_list):
        for i, pipeline in enumerate(pipeline_list):
            cross_val = cross_val_score(pipeline[1], self.features, self.targets, cv=5)
            population[i].fitness.values = (len(population[i]),cross_val.mean())
            # print(pipeline[0], cross_val.mean(), sep='\n')

    def _mate_operator(self, ind1, ind2):
        common_models = []
        for model_ind1 in ind1:
            for model_ind2 in ind2:
                if isinstance(model_ind1[1], type(model_ind2[1])):
                    common_models.append((model_ind1, model_ind2))
        model = random.choice(common_models)
        if model[0][1].type_model != 'classification':
            param = random.choice(list(model[0][1].__dict__.keys()))
            new_ind = deepcopy(ind1)
            setattr(new_ind[model[0][0]-1][1], param, getattr(ind2[model[1][0]-1][1],param))
            return new_ind
        param_model1 = model[0][1].__dict__
        param_model2 = model[1][1].__dict__
        param_grid = {}
        for key in param_model1.keys():
            if param_model1[key] == param_model2[key]:
                param_grid[key] = [param_model1[key]]
            else:
                param_grid[key] = [param_model1[key], param_model2[key]]
        param_grid = deepcopy(param_grid)
        if len(param_grid) == 2:
            return ind1
        else:
            param_grid.pop('type_model')
            param_grid.pop('name_model')
            t = GridSearchCV(self.model_to_sklearn(model[0][1],False), param_grid, cv=5,scoring="accuracy").fit(self.features, self.targets)
            new_params = t.best_params_
            new_params['type_model'] = model[0][1].type_model
            new_params['name_model'] = model[0][1].name_model
            # new_model = (model[0][0],type(model[0][1].name_model,(),new_params))
            new_ind = deepcopy(ind1)
            new_ind[model[0][0]-1][1].__dict__ =  new_params
            return new_ind




    def _mutate_operator(self, ind):
        if np.random.random() <= 1/3:
            new_ind = self.replacement_mutation(ind)
            self.add_info(new_ind)
        elif np.random.random() <= 2/3:
            new_ind = self.shrink_mutation(ind)
            self.add_info(new_ind)
        else:
            new_ind = self.insert_mutation(ind)
            self.add_info(new_ind)
        return new_ind

    def shrink_mutation(self, ind):
        if len(ind) > 1:
            del_model = random.choice(ind[:-1])
            for i in range(del_model[0],len(ind)):
                ind[i] = (ind[i][0]-1,ind[i][1])
            ind.pop(del_model[0]-1)
        return ind

    def insert_mutation(self, ind):
        CALLABLES = (types.FunctionType, types.MethodType)
        if len(ind) < self.max_height:
            primitive_obj = random.choice(self._base_primitives)()
            for key in [i for i in dir(primitive_obj) if
                        not i.startswith('__') and not isinstance(getattr(primitive_obj, i), CALLABLES)] :
                setattr(primitive_obj, key, random_value_from(primitive_obj, key))
            ind.insert(len(ind)-1,(len(ind), primitive_obj))
            ind[-1] = (ind[-1][0]+1,ind[-1][1])
        return ind

    def replacement_mutation(self, ind):
        ind_copy = deepcopy(ind)
        model = random.choice(ind_copy)
        if model[0] < len(ind) and np.random.random() <= 0.5:
            CALLABLES = (types.FunctionType, types.MethodType)
            primitive_obj = random.choice(self._base_primitives)()
            for key in [i for i in dir(primitive_obj) if
                        not i.startswith('__') and not isinstance(getattr(primitive_obj, i), CALLABLES)] :
                setattr(primitive_obj, key, random_value_from(primitive_obj, key))
            ind_copy[model[0]-1] = (model[0], primitive_obj)
        else:
            params = deepcopy(model[1].__dict__)
            params.pop('type_model')
            params.pop('name_model')
            if params != {}:
                param = random.choice(list(params.keys()))
                if model[0] == len(ind_copy):
                    k = random.choice(classification_models[model[1].name_model][param])
                else:
                    if model[1].type_model == 'selection':
                        k = random.choice(selection_models[model[1].name_model][param])
                    elif model[1].type_model == 'preprocessing':
                        k = random.choice(preprocessing_models[model[1].name_model][param])
                setattr(model[1], param, k)
        return ind_copy






    def add_info(self, ind):
        names = []
        for model in ind:
            names.append(model[1].name_model)
        ind.info = tuple(names)

    def get_random_two_ind_for_mate(self, population):
        favorable_inds = []
        for i in range(len(population)):
            for j in range(i+1,len(population)):
                a = set(population[i].info)
                b = set(population[j].info)
                if a.intersection(b) != set():
                    favorable_inds.append((population[i], population[j]))
        random_inds = random.choice(favorable_inds)
        return deepcopy(random_inds)

    def changes(self):
        population_copy = deepcopy(self.__population)
        offspring = []
        time_table = []
        i = 0
        while i < self.population_size:
            begin_change = time()
            if np.random.random() < self.probability_mate:
                type_change = 'mate'
                new_ind = self._toolbox.mate(*self.get_random_two_ind_for_mate(population_copy))
            elif np.random.random() <= self.probability_mate + self.probability_mutation:
                type_change = 'mutate'
                new_ind = self._toolbox.mutate(random.choice(population_copy))
            stop_change = time() - begin_change
            begin_check = time()
            count = 0
            for ind in offspring:
                if self.compare_inds(ind, new_ind):
                    count += 1
            for ind in self.__population :
                if self.compare_inds(ind, new_ind) :
                    count += 1
            stop_check = time() - begin_check
            if count < 5 :
                time_table.append([i,type_change,stop_change,stop_check])
                offspring.append(new_ind)
                i += 1
        return offspring

    def compare_inds(self, ind1, ind2):
        if len(ind1) != len(ind2):
            return False
        for i in range(len(ind1)):
            if not isinstance(ind1[i][1], type(ind2[i][1])):
                return False
        return True



    def _pre_init_fit(self):
        self.__population = []
        self.create_base_primitives_and_terminals()
        self._setup_toolbox()
        self.__population = self._toolbox.population()

    def fit(self, features, targets):
        self.features = features
        self.targets = targets
        self.base_pop = []
        self._pre_init_fit()
        for ind in self.__population:
            self.add_info(ind)
        for _ in range(self.generations):
            print([_]*200)
            self.pipeline_list = [(individual, self.individual_to_pipeline(individual)) for individual in
                                  self.__population]
            print('Началась оценка')
            s = time()
            self._evaluation_individuals(self.__population, self.pipeline_list)
            print('Произошла оценка ',time()-s)
            if _ == 3:
              self.old_pop = deepcopy(self.__population)
            s = time()
            offspring = self.changes()
            print('Потомки созданы ',time()-s)
            # for ind in self.__population:
            #     print('pop',ind)
            # for ind in offspring:
            #     print('off', ind)
            self.pipeline_list = [(individual, self.individual_to_pipeline(individual)) for individual in
                                  offspring]
            print('Началась оценка')
            s = time()
            self._evaluation_individuals(offspring, self.pipeline_list)
            print('Произошла оценка ',time()-s)
            self.base_pop.append((deepcopy(self.__population),deepcopy(offspring)))
            self.__population[:] = self._toolbox.select(self.__population+offspring,30)




        # self.__population[:] = self._toolbox.select(self.__population,5)
        # self._mate_operator(*self.get_random_two_ind_for_mate())
        return self.__population

class GeneticClassification(GeneticBase):
    def __init__(self, generations=100, population_size=100, offspring_size=None):
        super().__init__(generations,population_size, offspring_size)
        self.type = 'Classification'




if __name__ == '__main__' :
    df = pd.read_csv('../datasets/sonar.csv')
    # df = df.drop(df.index[150 :])
    x_train, x_test, y_train, y_test = train_test_split(df.drop(df.columns[-1], 1), df[df.columns[-1]], test_size=.2,
                                                        random_state=42)
    print(df)
    GB = GeneticBase(population_size=30,generations=5)
    GB.fit(x_train,y_train)