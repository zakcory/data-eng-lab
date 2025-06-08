import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import argparse
from matplotlib import pyplot as plt


class ActiveLearningPipeline:
    def __init__(self, seed,
                 test_indices,
                 available_pool_indices,
                 train_indices,
                 selection_criterion,
                 iterations=10,
                 budget_per_iter=30,
                 nodes_df_path='nodes.csv',
                 subject_mapping_path='subject_mapping.pkl'):
        self.seed = seed
        self.iterations = iterations
        self.budget_per_iter = budget_per_iter
        self.nodes_df_path = nodes_df_path
        self.available_pool_indices = available_pool_indices
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.selection_criterion = selection_criterion
        self.nodes_df = pd.read_csv(nodes_df_path)
        self.node_id_mapping = self._get_node_id_mapping()
        self.feature_vectors = self._read_feature_vectors()
        self.labels = self._read_labels(subject_mapping_path)
        # TODO: Add (if needed) additional fields and functions to the constructor.
        # TODO: Complete the implementation of the run_pipeline method (note: it should be called externally,
        #  not from within the constructor).
        # TODO: Implement the custom sampling method (you may modify its arguments if needed).
        # TODO: You may add helper class methods, but no method should exceed 15 lines of code.
        # TODO: Do not modify the constructor signature or any already implemented methods.

    def _get_node_id_mapping(self):
        """
        Get node id mapping dictionary (from row index to node id) according to the nodes dataframe
        :return:
        self.node_id_mapping: dictionary, node id to row index
        """
        node_id_mapping = {}
        for idx, row in self.nodes_df.iterrows():
            node_id = row['nodeId']
            node_id_mapping[node_id] = idx
        return node_id_mapping

    def _read_feature_vectors(self):
        """
        Read feature vectors from the nodes dataframe
        :return:
        feature_vectors: numpy array, feature vectors
        """
        feature_vectors_raw = self.nodes_df['features'].apply((lambda x: x.strip('][').split(', ')))
        return np.array([[float(val) for val in feature_vector] for feature_vector in feature_vectors_raw])

    def _read_labels(self, subject_mapping_path):
        """
        Read subjects from the nodes dataframe, and convert them to labels (integers)
        :return:
        labels: numpy array, labels
        """
        with open(subject_mapping_path, 'rb') as f:
            subject_mapping = pickle.load(f)
        labels = self.nodes_df['subject'].apply(lambda x: subject_mapping[x])
        return np.array(labels)

    def run_pipeline(self):
        """
        Run the active learning pipeline
        """
        # TODO: Implement the active learning pipeline
        # TODO: Do not change the lines that are already implemented here in this method, only add your own code lines
        #  before and after them.
        accuracy_scores = []
        for iteration in range(self.iterations):
            if len(self.train_indices) > 600:
                # raise error if the train set is larger than 600 samples
                raise ValueError('The train set is larger than 600 samples')
            print(f'Iteration {iteration + 1}/{self.iterations}')
            trained_model = self._train_model()
            accuracy = self._evaluate_model(trained_model)
            accuracy_scores.append(accuracy)
            print(f'Accuracy: {accuracy}')
            print('----------------------------------------')
        return accuracy_scores

    def _train_model(self):
        """
        Train the model
        """
        model = RandomForestClassifier(random_state=self.seed)
        train_indices = [self.node_id_mapping[node_id] for node_id in self.train_indices]
        return model.fit(self.feature_vectors[train_indices], self.labels[train_indices])

    def _random_sampling(self):
        """
        Random samplings
        :return:
        new_selected_samples: numpy array, new selected samples
        """
        np.random.seed(self.seed)
        return np.random.choice(list(range(len(self.available_pool_indices))), self.budget_per_iter, replace=False)

    def _custom_sampling(self):
        """
        Custom sampling method to be implemented
        :return:
        new_selected_samples: numpy array, new selected samples
        """
        # todo: implement custom sampling
        pass

    def _update_train_indices(self, new_selected_samples):
        """
        Update the train indices
        """
        self.train_indices = np.concatenate([self.train_indices, new_selected_samples])

    def _update_available_pool_indices(self, new_selected_samples):
        """
        Update the available pool indices
        """
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, new_selected_samples)

    def _evaluate_model(self, trained_model):
        """
        Evaluate the model
        :param trained_model: trained model
        :return: accuracy: float, accuracy of the model on the test set
        """
        test_indices = [self.node_id_mapping[node_id] for node_id in self.test_indices]
        train_indices = [self.node_id_mapping[node_id] for node_id in self.train_indices]
        if any(idx in train_indices for idx in test_indices):
            raise ValueError('Data leakage detected: test indices are in the train set.')
        preds = trained_model.predict(self.feature_vectors[test_indices])
        return round(np.mean(preds == self.labels[test_indices]), 3)


def generate_plot(accuracy_scores_dict):
    num_iters = len(accuracy_scores_dict['random'])
    for criterion, accuracy_scores in accuracy_scores_dict.items():
        x_vals = list(range(1, len(accuracy_scores) + 1))
        plt.plot(x_vals, accuracy_scores, label=criterion)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.xlim(1, num_iters)
    plt.xticks(range(1, num_iters + 1))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices_dict_path', type=str, default='indices_dict_part1.pkl')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--budget_per_iter', type=int, default=30)
    parser.add_argument('--nodes_df_path', type=str, default='nodes.csv')
    parser.add_argument('--subject_mapping_path', type=str, default='subject_mapping.pkl')

    hp = parser.parse_args()
    with open(hp.indices_dict_path, 'rb') as f:
        indices_dict = pickle.load(f)
    available_pool_indices = indices_dict['available_pool_indices']
    train_indices = indices_dict['train_indices']
    test_indices = indices_dict['test_indices']

    selection_criteria = ['custom', 'random']
    accuracy_scores_dict = defaultdict(list)

    for seed in range(1, 4):
        print(f"seed {seed}")
        for criterion in selection_criteria:
            AL_class = ActiveLearningPipeline(seed=seed,
                                              test_indices=test_indices,
                                              available_pool_indices=available_pool_indices,
                                              train_indices=train_indices,
                                              selection_criterion=criterion,
                                              iterations=hp.iterations,
                                              budget_per_iter=hp.budget_per_iter,
                                              nodes_df_path=hp.nodes_df_path,
                                              subject_mapping_path=hp.subject_mapping_path)
            accuracy_scores_dict[criterion] = AL_class.run_pipeline()
        generate_plot(accuracy_scores_dict)
        print(f"======= Finished iteration for seed {seed} =======")
