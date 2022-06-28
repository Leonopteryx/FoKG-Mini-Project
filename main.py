from rdflib import Namespace, Graph, Literal, URIRef
from ontolearn.base import KnowledgeBase
from random import shuffle
import numpy as np
from ontolearn.concept import Concept
import time
start_time = time.time()
print("Started--- %s seconds ---" % (start_time))

NS_CAR = Namespace("http://dl-learner.org/carcinogenesis#")
NS_RES = Namespace("https://lpbenchgen.org/resource/")
NS_PROP = Namespace("https://lpbenchgen.org/property/")
NS_RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
NS_CLASS = Namespace("https://lpbenchgen.org/class/")


class Model:

    def __init__(self, ontology_path):
        self.kb = KnowledgeBase(path=ontology_path)
        self.all_instances = self.__setup()

    def __setup(self):
        """
        Get all instances from knowledge base
        :return: instances of the kb
        """

        instances = set()
        for c in self.kb.concepts.values():
            c.instances = {jjj for jjj in c.owl.instances(world=self.kb.onto.world)}
            instances.update(c.instances)
        self.kb.thing.instances = instances
        self.kb.thing.owl.namespace = self.kb.onto.get_namespace("http://dl-learner.org/carcinogenesis#")
        return instances

    def cross_validation(self, lp, folds=10):
        """
        Perform a cross validation on the specified lp and with the specified folds.
        For training data ( for first file of lps)
        :param lp: Learning problem
        :param folds: Folds of the cross validation
        :return: Mean F1-score
        """
        print("Current LP: ", lp['name'])

        pos, neg = self._get_objects_for_iris(lp['pos'], lp['neg'])
        pos = list(pos)
        neg = list(neg)
        shuffle(pos)
        shuffle(neg)
        pos_folds = np.array_split(pos, folds)
        neg_folds = np.array_split(neg, folds)

        f_1 = 0
        for val in range(folds):
            pos_train = np.concatenate(pos_folds[:val] + pos_folds[val + 1:])
            neg_train = np.concatenate(neg_folds[:val] + neg_folds[val + 1:])
            pos_val = pos_folds[val]
            neg_val = neg_folds[val]

            solution = self._run_algorithm(set(pos_train), set(neg_train))

            val_score_f1 = self._f1(solution[0], set(pos_val), set(neg_val))
            print("Fold ", val + 1, " : ", solution[0].str,
                  " - Val score F1: ", val_score_f1,
                  "Train score F1: ", solution[1])

            f_1 += val_score_f1

        print('Mean Score: ', f_1 / folds, "\n\n")

    def fit_and_predict(self, lp):
        """
        Trains the model with the given lp (pos and neg) and classifies the
        remaining instances ( for second file of lps)
        :param lp: Learning problem
        :return: Positive and negative classification of the remaining instances
        """

        print("Current LP: ", lp['name'])
        pos = lp['pos']
        neg = lp['neg']

        pos_ind, neg_ind = self._get_objects_for_iris(pos, neg)

        solution = self._run_algorithm(pos_ind, neg_ind)
        concept = solution[0]
        print("Found solution: ", concept.str, ' - Train Score: ', solution[1])

        pos_and_neg = pos_ind | neg_ind
        remaining_instances = {inst for inst in self.all_instances if inst not in pos_and_neg}

        pos_classified = concept.instances & remaining_instances
        neg_classified = remaining_instances - pos_classified

        pos_classified = [p.iri for p in pos_classified]
        neg_classified = [n.iri for n in neg_classified]

        print("Classified - Positives: ", len(pos_classified), " Negatives: ", len(neg_classified), "\n")

        return pos_classified, neg_classified

    def _run_algorithm(self, pos, neg):
        """
        Returns the best concept for the given set of pos and neg examples.
        :param pos: Positive examples
        :param neg: Negative examples
        :return: Best concept
        """

        current_best = (self.kb.thing, self._f1(self.kb.thing, pos, neg))
        concepts = set()
        for c in self.refine_thing():
            score = self._f1(c, pos, neg)
            if score > 0:
                concepts.add(c)
            if score == 1.0:
                return c, score
            if self._compare_concepts((c, score), current_best):
                current_best = (c, score)

        for _ in range(3):
            temp = set()
            for c in concepts:
                refinements = self.refine(c)

                scores = [(c, self._f1(c, pos, neg)) for c in refinements]
                # Find max in F1-Score, tiebreak on length
                best = max(scores, key=lambda c: (c[1], -c[0].length))

                if best[1] == 1.0:
                    return best[0], best[1]
                if best[1] > 0:
                    temp.add(best[0])
                if self._compare_concepts(best, current_best):
                    current_best = best
            concepts = temp

        return current_best

    def _f1(self, concept, pos, neg):
        """
        :param concept: Concept to be tested
        :param pos: Positive examples
        :param neg: Negative examples
        :return: F1-score of the given concept on the given pos and neg examples
        """

        instances = concept.instances
        tp = len(instances & pos)
        fp = len(instances & neg)
        fn = len(pos - instances)
        f1 = tp / (tp + 0.5 * (fp + fn))
        return round(f1, 3)

    def _compare_concepts(self, c1, c2):
        """
        Compares the given concepts depending on F1-score. If they have the same quality
        length will be taken into consideration.
        :param c1: First concept - Structure containing the concept and the quality(F1-score)
        :param c2: Second concept - Structure containing the concept and the quality(F1-score)
        :return: True if 1st concept has better quality than 2nd concept or is shorter if they are the same quality,
        else false.
        """

        return c1[1] > c2[1] or (c1[1] == c2[1] and c1[0].length <= c2[0].length)

    def _get_objects_for_iris(self, pos, neg):
        """
        Searches for the IRIs of pos and neg examples and build a set with the respective individual.
        :param pos: Positive examples as IRIs
        :param neg: Negative examples as IRIs
        :return: Set of pos and neg examples as individual of kb
        """

        pos_ind = {self.kb.onto.search(iri=p)[0] for p in pos}
        neg_ind = {self.kb.onto.search(iri=n)[0] for n in neg}
        return pos_ind, neg_ind

    def refine(self, concept):
        """
        Refines the concept
        :param concept: Concept to be refined
        :return: Refinements set
        """

        if concept == self.kb.thing:
            return self.refine_thing()
        elif concept.is_atomic:
            return self.refine_atomic(concept)
        elif concept.form == "ObjectSomeValuesFrom":
            return self.refine_existential_restriction(concept)
        elif concept.form == "ObjectIntersectionOf":
            return self.refine_intersection(concept)
        else:
            raise ValueError

    def refine_thing(self):
        """
        Refines the top concept T to sub-concepts for each object property
        :return: Refinements set
        """
        thing = self.kb.thing
        refinements = set()
        refinements.update(self.kb.top_down_direct_concept_hierarchy[thing])
        for p in self.kb.property_hierarchy.object_properties:
            refinements.add(self.kb.existential_restriction(thing, p))
        return refinements

    def refine_atomic(self, concept):
        """
        Refines concept to sub-concepts and conjunction of C with T
        :param concept: Atomic concept to be refined
        :return: Refinements set
        """
        refinements = set()
        refinements.update(self.kb.top_down_direct_concept_hierarchy[concept])
        refinements.add(self.kb.intersection(concept, self.kb.thing))
        return refinements

    def refine_existential_restriction(self, concept):
        """
        Refines existential restriction
        :param concept:
        :return: Refinements set
        """
        return {self.kb.existential_restriction(ref, concept.role)
                for ref in self.refine(concept.filler)}

    def refine_intersection(self, concept):
        """
        Refines the conjunction
        :param concept: Concept to be refined
        :return: Refinements set
        """
        refinements = set()
        refinements.update({self.kb.intersection(concept.concept_a, r)
                            for r in self.refine(concept.concept_b)})
        refinements.update({self.kb.intersection(r, concept.concept_b)
                            for r in self.refine(concept.concept_a)})
        return refinements


def create_lps_list(path):
    """
    Uses the path to get the learning problems and add them on a list.
    :param path: Path of the lps ttl file.
    :return: List that have each lp saved as a dictionary of keys "name", "pos" and "neg".
    """
    lp_instance_list = []
    with open(path, "r") as lp_file:
        for line in lp_file:
            if line.startswith("lpres:"):
                lp_key = line.split()[0].split(":")[1]
            elif line.strip().startswith("lpprop:excludesResource"):
                exclude_resource_list = line.strip()[23:].split(",")
                exclude_resource_list = [individual.replace(";", "")
                                             .replace("carcinogenesis:",
                                                      "http://dl-learner.org/carcinogenesis#").strip()
                                         for individual in exclude_resource_list]
            elif line.strip().startswith("lpprop:includesResource"):
                include_resource_list = line.strip()[23:].split(",")
                include_resource_list = [individual.replace(".", "")
                                             .replace("carcinogenesis:",
                                                      "http://dl-learner.org/carcinogenesis#").strip()
                                         for individual in include_resource_list]
                lp_instance_list.append({"name": lp_key,
                                         "pos": include_resource_list,
                                         "neg": exclude_resource_list})

    return lp_instance_list


def add_lp_to_graph(graph, lp_name, pos, neg, index):
    """ Add the given learning problem together with positive and negative classified
        individuals for that lp to the graph (Format as specified in the slides)
    Args:
        Learning problem name, positive and negative classifications
        for that learning problem
    """
    current_pos = f'result_{index}pos'
    current_neg = f'result_{index}neg'
    graph.add((URIRef(NS_RES + lp_name), NS_RDF.type, NS_CLASS.LearningProblem))
    
    for p in pos:
        graph.add((URIRef(NS_RES + lp_name), NS_PROP.includesResource, URIRef(p)))


def run():
    """
    Runs the algorithm on the test dataset. It trains on the learning problems
    provided and classifies the remaining instances.
    """
    lps = create_lps_list('data/kg22-carcinogenesis_lps2-test.ttl')
    model = Model(ontology_path="data/carcinogenesis.owl")
    
    g = Graph()
    g.bind('lpclass', NS_CLASS)
    g.bind('carcinogenesis', NS_CAR)
    g.bind('rdf', NS_RDF)
    g.bind('lpres', NS_RES)
    g.bind('lpprop', NS_PROP)

    for idx, lp in enumerate(lps):
        pos, neg = model.fit_and_predict(lp)
        add_lp_to_graph(g, lp['name'], pos, neg, idx + 1)

    g.serialize(destination='classification_result.ttl', format='turtle')
    
    
    print("Completed--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    run()