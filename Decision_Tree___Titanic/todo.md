# Deliverables

- Do EDA + preprocessing requried
- Best Fitting decision tree using hyper-parameter tuning (GridSearchCV)
- Explore the ensembles- Random Forest and XG Boost ( compare to single decision tree )

# Note
- Include all the steps
- feature engineering
- Well Presented

Do all the preprocessing and EDA as required.
2. Find the best fitting decision tree using hyper-parameter tuning
(GridSearchCV)

Make sure you make it was PRESENTABLE and PRETTY ( lots of graphs and statistics displayed wherever you thing it makes the ipynb notebook prettier )

sample train.csv :-
406,0,2,"Gale, Mr. Shadrach",male,34.0,1,0,28664,21.0,,S
743,1,1,"Ryerson, Miss. Susan Parker ""Suzette""",female,21.0,2,2,PC 17608,262.375,B57 B59 B63 B66,C
261,0,3,"Smith, Mr. Thomas",male,,0,0,384461,7.75,,Q
368,1,3,"Moussa, Mrs. (Mantoura Boulos)",female,,0,0,2626,7.2292,,C
159,0,3,"Smiljanic, Mr. Mile",male,,0,0,315037,8.6625,,S
555,1,3,"Ohman, Miss. Velin",female,22.0,0,0,347085,7.775,,S
830,1,1,"Stone, Mrs. George Nelson (Martha Evelyn)",female,62.0,0,0,113572,80.0,B28,
678,1,3,"Turja, Miss. Anna Sofia",female,18.0,0,0,4138,9.8417,,S
381,1,1,"Bidois, Miss. Rosalie",female,42.0,0,0,PC 17757,227.525,,C


this is from the titanic data set with the dataset description being


Data Dictionary

Variable Definition Key
survival Survival 0 = No, 1 = Yes
pclass Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
sex Sex
Age Age in years
sibsp # of siblings / spouses aboard the Titanic
parch # of parents / children aboard the Titanic
ticket Ticket number
fare Passenger fare
cabin Cabin number
embarked Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

Variable Notes

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way…
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way…
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them. 

In Preprocessing methode can you add a newFeature called is_Alone that is equal to 1 if parch=0 and sibsp=0

can you please JUST FILL THIS CODEi

# Feature engineering ; additional DOMAIN SPECIFIC features and 
for dataset in combine:
    dataset['FamilySize']     = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['FamilySize_and_Age'] = dataset['FamilySize'] * dataset['Age']
    dataset['Pclass_Fare'] = dataset['Pclass'] * dataset['Fare']
    dataset['Pclass_Age'] = dataset['Pclass'] * dataset['Age']
    dataset['Fare_per_Person'] = dataset['Fare'] / dataset['FamilySize']
    dataset['is_Alone'] = ...
