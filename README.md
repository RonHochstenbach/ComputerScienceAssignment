# ComputerScienceAssignment
This code is used for the assignment for the course Computer Science (FEM21037) at Erasmus University Rotterdam. It is structured as follows:

Lines 52 - 145 contain auxiliary functions making train-test splits and calculating similarities

Lines 244 - 291 import the data, a json file of tvs from multiple webshops. It also performs rudimentary data analysis

Lines 294 - 451 provide 2 data cleaning functions, first only homogenizing units, then also converting fractions to decimals.

Lines 459 - 531 provide 2 functions to create model words and add representations based on these to data points

Lines 453 - 609 contain a function that performs Locality Sensitive Hashing to an input data set

Lines 618 - 1070 provide 3 methods for doing final classification. final_classification_base only deletes options based on brand and shops, final_classification_base adds to this a similarity measure with threshold, and final_classcluster adds clustering.  class_qgram_hyperopt and lusterclass_hyperopt adapt the latter two to account for hyperparameter optimization.

Lines 1073 - 1189 can be used to run an algorithm for a certain number of bootstraps.

Lines 1189 - 1308 Can be used to perform hyperparameter optimization

Lines 1308 - 1509 Cycle over bands to make graphs

Lines 1513 - 1530 can be used to make the graphs
