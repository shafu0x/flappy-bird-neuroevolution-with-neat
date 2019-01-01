import numpy as np
import datetime
import os

from network import NN

class NEAT:
    def __init__(self, population_size, n_parents_new_gen, n_mutated_parents_new_gen, n_mates_new_gen, prob_weight_change, gamma):
        self.population_size = population_size
        self.n_parents_new_gen = n_parents_new_gen
        self.n_mates_new_gen = n_mates_new_gen
        self.n_mutated_parents_new_gen = n_mutated_parents_new_gen
        self.prob_weight_change = prob_weight_change
        self.gamma = gamma
        self.n_gen = 0
        self.nn_id = 1
        self.score_file = open('scores.txt', 'w+')
        now = datetime.datetime.now()
        self.current_directory = '{}-{}-{}-{}'.format(now.day, now.hour, now.minute, now.second)
        os.mkdir('trained_nn/{}'.format(self.current_directory))
        self.population = self.init_population()

    def init_population(self):
        initial_population = {}
        for i in range(self.population_size):
            nn = NN()
            initial_population[self.nn_id] = [self.n_gen, nn, 0] # 0 is the score
            self.nn_id += 1
        return initial_population

    def init_new_gen(self, rated_population, is_first_new_gen=False):
        new_population = {}
        if is_first_new_gen:
            print(type(self.population))
            for k, v in self.population.iteritems():
                nn = v[1]
                if nn.is_average_prediction_good(0.5):
                    new_population[k] = v
            print('init pop size is {}'.format(len(new_population)))
            self.population = new_population
            return

        sorted_population_by_score = sorted(self.population.items(), key=lambda x: x[1][2], reverse=True)
        self.save_nn(sorted_population_by_score[0][1][1], sorted_population_by_score[0][1][2]) # save the best nn
        print('###############################')
        print('GEN # ', self.n_gen)
        print('NN with non zero score: ', str(self.get_n_nn_with_non_zero_score(sorted_population_by_score)))
        print('###############################')
        self.score_file.write('GEN {}: highest score: {}'.format(self.n_gen, sorted_population_by_score[0][1][2]))
        self.n_gen += 1

        for i in range(self.n_mates_new_gen):
            parent_nn_1 = sorted_population_by_score[i][1][1]
            parent_nn_2 = sorted_population_by_score[i+1][1][1]
            child_nn = self.mate(parent_nn_1, parent_nn_2)
            self.nn_id += 1
            new_population[self.nn_id] = [self.n_gen, child_nn, 0]

        for i in range(self.n_mutated_parents_new_gen):
            self.nn_id += 1
            old_good_nn_id = sorted_population_by_score[i][0]
            old_good_nn_gen = sorted_population_by_score[i][1][0]
            old_good_nn = sorted_population_by_score[i][1][1]
            new_population[self.nn_id] = [old_good_nn_gen, self.mutate(old_good_nn), 0]

        for i in range(self.n_parents_new_gen):
            old_good_nn_id = sorted_population_by_score[i][0]
            old_good_nn_gen = sorted_population_by_score[i][1][0]
            old_good_nn = sorted_population_by_score[i][1][1]
            new_population[old_good_nn_id] = [old_good_nn_gen, old_good_nn, 0]

        self.population = new_population

    def save_nn(self, nn, score):
        now = datetime.datetime.now()
        name = 'NN_gen_{}_from_{}-{}-{}-{}'.format(self.n_gen, now.day, now.hour, now.minute, now.second)
        nn.model.save('{}/{}/{}.h5'.format('trained_nn',self.current_directory, name))

    def set_score(self, nn_id, score):
        self.population[nn_id][2] = score

    def mate(self, nn_1, nn_2):
        nn_1_weight_1 = nn_1.model.layers[:][1].get_weights()
        nn_2_weight_0 = nn_2.model.layers[:][0].get_weights()

        nn = NN()
        nn.model.layers[:][0].set_weights(nn_2_weight_0)
        nn.model.layers[:][1].set_weights(nn_1_weight_1)
        return nn

    def mutate(self, nn):
        layers = nn.model.layers[:]
        for l, layer in enumerate(layers):
            new_weights = []
            weights = layer.get_weights()[0]
            for i in range(weights.shape[0]):
                for weight in weights[i]:
                    rand = np.random.random() #[0,1]
                    if self.prob_weight_change/100.0 > rand:
                        update = self.gamma * np.random.normal()
                        weight += update
                        weight = np.round(weight, 5)
                    new_weights.append(weight)

            layer_weights = []
            new_weights = np.array(new_weights).reshape(weights.shape[0],weights.shape[1])
            layer_weights.append(new_weights)
            layer_weights.append(layer.get_weights()[1])

            layers[l].set_weights(np.array(layer_weights))
        return nn

    def get_n_nn_with_non_zero_score(self, sorted_population):
        i = 0
        for p in sorted_population:
            if p[1][2] > 0:
                i += 1
        return i

    def remove_nn_from_population(self, id):
        self.population.pop(id)
