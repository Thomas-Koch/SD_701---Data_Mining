---------------------------
# <center>-- SD 701 : Data Mining Project --</center>
-------------
    
         
***Philippe Bénézeth          
et           
Thomas Koch***

------------------------
## Préambule

Ce projet s'inscrit dans la continuité de ce qui a été vu en cours durant le premier TP, à savoir comment faire de la recommandation de films. Nous avons donc repris la base de données ***movie lens*** pour réaliser notre projet. Notre idée était de trouver deux nouvelles approches pour effectuer de la recommandation et de tenter une comparaison entre ces deux approches. Les codes qui nous ont permis d'arriver à nos résultats se trouvent dans les notebook 1 et 2.


## Première approche "KNN" : approche centrée sur la collaboration

### Explications sur le modèle

Pour cette première approche, nous sommes partis du fait que chaque film était associé à un ou plusieurs genres. Au total, il existe **20 "genres" différents**.


L'idée de départ est de considérer les notes moyennes des utilisateurs selon les genres des films et d'établir ainsi un "profil" d'utilisateur. Nous avons donc calculé un profil d'utilisateur qui correspond à un vecteur de 20 coordonnées représentant la moyenne des notes attribuées par cet utilisateur aux films selon leurs genres.

Pour un film donné, le principe est alors de faire l'hypothèse que 2 utilisateurs ayant des sous-profils (variables limitées aux genres du film) proches (au sens de la distance euclidienne) noteront le film de façon similaire. Voilà donc pourquoi nous avons utilisé une approche KNN, en prenant ici un nombre de voisins équivalent à k = 5.

Afin d'avoir des résultats intéressants basés sur des datasets suffisamment importants, nous avons limité les prédictions sur les 19 users qui ont vu plus de 250 films parmi ceux vus par plus de 50 users. Ainsi, pour chaque film vu par le user considéré et pour lequel nous avons au moins 50 notes dans la base de notations, nous considérons les genres qui lui sont attachés. Nous lançons ensuite le KNN sur une projection de profil sur ces genres. Nous moyennons les notes des 5 voisins sur le film pour prédire la note du user considéré.

Ayant remarqué un biais important entre la note moyenne mise par un utilisateur sur la totalité des films vus et la note moyenne de l'ensemble des autre utilisateurs sur la totalité des films nous avons choisi de dé-biaiser la note moyenne obtenue sur les voisins du KNN.

>Les notes allant de 0.5 étoile à 5 étoiles, Nous avons considéré la prédiction réussie lorsqu'elle se situe à 1/2 étoile près. Pour la prédiction réussie, on affiche un résultat de **25.7%** (à comparer aux 10% d'une prédiction aléatoire). Pour la prédiction à une 1/2 étoile près, on ajoute **38.6%** soit un **total de 64.3%** de réussite cumulée (à comparer aux 30% d'une prédiction aléatoire).

### Discussion sur le modèle et ses résultats

Même si les performances de ce modèle sont relativement bonnes, nous avons constaté des problèmes importants, que sont :
>* les "démarrages à froid", ou comment faire des recommandations pour des nouveaux utilisateurs n'ayant pas assez évalués de films
>* la nécessité d'avoir beaucoup de données pour faire des prédictions correctes 
>* le biais de popularité, ou comment faire pour recommander des films peu notés
>* le goulot d'étranglement

Il existe globalement trois cas de démarrage à froid:
>* **Nouvelle communauté :** début du système de recommandation avec un catalogue d'articles mais  catalogue de films mais presque aucun utilisateur et interaction entre utilisateurs pour réussir à fournir des recommandations fiables.
>* **Nouveau film :** un nouveau film est ajouté au système, il peut il y avoir quelques informations sur son genre par exemple mais aucune évaluation d'utilisateur.
>* **Nouvel utilisateur :** un nouvel utilisateur s'inscrit et n'a pas encore évalué de films, il n'est donc pas possible de fournir des recommandations personnalisées.

Le dernier problème n'est ici pas le plus gênant pour nous dans le sens où nous pouvons utiliser le filtrage par films pour faire des recommandations à un nouvel utilisateur. Elles ne seront certes pas personnalisées mais pourrons tout de même avoir du sens. En revanche, les deux premiers cas sont plus problématiques pour nous, en particulier le second.

Pour le démarrage à froid des films, cela constitue un problème pour les algorithmes de filtrage collaboratif principalement du fait qu'ils s'appuient sur les interactions de l'élément pour faire des recommandations. Si aucune interaction n'est disponible, un algorithme collaboratif pur ne peut pas recommander le film. Dans le cas où seules quelques interactions sont disponibles, bien qu'un algorithme collaboratif puisse le recommander, la qualité de ces recommandations sera médiocre. 

Ce qui mène donc à un autre problème, non plus lié aux nouveaux éléments, mais plutôt aux éléments impopulaires. Dans le cas des recommandations de films, il peut arriver qu'une poignée de films reçoivent un nombre extrêmement élevé d'évaluation, alors que la majorité des autres n'en reçoivent qu'une fraction. C'est ce qui est appelé biais de popularité. 

En plus de cela, l'évolutivité ou scalabilité constitue un gros point faible du modèle KNN. La latence de prédiction peut être importante avec de gros volumes de données comme nous avons pu nous en rendre compte durant nos tests.

### Proposition de solution

Afin de surmonter ces problèmes, le système peut décider de ne pas émettre de recommandations à un utilisateur dont le système n’a pas atteint un certain nombre d’informations
requises. Toutefois, cette pratique limite l’efficacité du système et est à éviter. Une solution plus avantageuse à la fois pour l’utilisateur et pour le système de recommandations est d’utiliser d’autres sources d’informations. En ayant ainsi un système hybride, par exemple un système utilisant à la fois le filtrage collaboratif et le filtrage par contenu, il est possible d’aider l’utilisateur à faire un choix judicieux en lui offrant des recommandations, qu'il soit nouveau ou non. Le système utilisera de l’information basée
sur le contenu jusqu’à temps que l’usager ou le film atteignent un certain nombre
d'évaluations, puis basculera sur une approche collaborative.


## Deuxième approche "ALS" : approche centrée sur le contenu

Après avoir testé notre approche KNN, nous avons rapidement identifié trois limites évidentes à celle-ci :
>* Le **biais de popularité**
>* Le **problème de démarrage à froid des films**
>* Le **problème d'évolutivité** si les données d'entraînement sont trop volumineuses.

Les trois problèmes ci-dessus sont des défis très typiques pour dans les systèmes recommandations collaboratifs. Ils surviennent naturellement avec la matrice d'interaction utilisateur-film (ou film-utilisateur) où chaque entrée enregistre une interaction d'un utilisateur i et d'un film j. Dans la "vraie vie", la grande majorité des films ne reçoivent que très peu, voire pas du tout, de notes de la part des utilisateurs. Nous devons donc étudier une matrice extrêmement clairsemée, ou sparse, avec de nombreuses entrées auxquelles il manque des valeurs.

Nous allons donc voir comment traiter cela avec le modèle ALS

### Explications sur le modèle Alternating Least Squares (ALS) de Spark ML

L'alternance des moindres carrés (ALS) est un algorithme de factorisation matricielle et fonctionne de manière parallèle. L'objectif de la factorisation de matrice est de minimiser l'erreur entre les vraies évaluations et les évaluations prédites. ALS est implémentée dans Apache Spark ML et est conçue pour résoudre les problèmes de filtrage collaboratif à grande échelle. ALS permet de résoudre les problèmes d'évolutivité et la rareté des données d'évaluations, tout en s'adaptant très bien aux grands jeux de données.


Sa routine d'entraînement minimise alternativement deux fonctions de perte. Il maintient d'abord la matrice utilisateur fixe et exécute la descente de gradient avec la matrice de films. Puis il maintient la matrice de films fixe et exécute une descente de gradient avec la matrice utilisateur. Sa bonne scalabilité vient du fait qu'ALS exécute sa descente de gradient en parallèle sur plusieurs partitions des données d'entraînement.


Tout comme les autres algorithmes d'apprentissage automatique, ALS possède son propre ensemble d'hyper-paramètres. Nous avons choisi de régler ses hyper-paramètres avec une "cross-validation" ou validation croisée. Les hyper-paramètres les plus importants dans ALS sont :
>* **maxIter :** le nombre maximum d'itérations à exécuter (par défaut à 10)
>* **rang :** le nombre de facteurs latents dans le modèle (par défaut à 10)
>* **regParam :** le paramètre de régularisation de l'ALS (par défaut à 1.0)

Le réglage des hyper-paramètre a été codé dans une fonction pour accélérer les itérations de réglage de notre validation croisée. La fonction en question est la suivante :
```python
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    train_data: spark DF with columns ['userId', 'movieId', 'rating']
    
    validation_data: spark DF with columns ['userId', 'movieId', 'rating']
    
    maxIter: int, max number of learning iterations
    
    regParams: list of float, one dimension of hyper-param tuning grid
    
    ranks: list of float, one dimension of hyper-param tuning grid
    
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        print("\n")
        for reg in regParams:
            # get ALS model
            als = ALS(userCol = "userId", itemCol = "movieId", ratingCol = "rating", coldStartStrategy = "drop", nonnegative=True) \
						.setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            
			# train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization))
    return best_model

```

>Concernant les résultats obtenus, nous sommes parvenus à environ **64.2 % de prédictions justes** sur notre échantillon de test, échantillon obtenu avec un randomSplit 80%-20% sur le jeu de départ :
```python
(training,test) = df_ratings2.randomSplit([0.8, 0.2])
```
>En prenant un jeu de données de test comprenant uniquement les 105 utilisateurs ayant vus plus de 250 films et les 450 films vus par plus de 50 utilisateurs, notre score tombe à **50.3% de prédictions correctes** (*i.e.* à plus ou moins 0.5 étoile près).

### Discussion sur le modèle et ses résultats 

Dans le cas d'un `randomSplit` classique, **nos résultats se rapprochent des résultats de obtenus avec le KNN**. Toutefois, comme expliqué précédemment, il faut tenir compte du fait que le **résultat obtenu avec l'ALS est un résultat représentatif de tout le dataset**, ce qui n'est pas tout à fait le cas de la méthode KNN, capable de réaliser un bon score avec des utilisateurs ayant notés beaucoup de films (plus de 250) et avec les films les plus regardés.

Dans le deuxième cas de test du modèle ALS, le résultat moins bon s'explique par le fait que nous avons dû **entraîner notre matrice sur une matrice encore plus sparse qu'initialement**, en **nous privant** qui plus est **des observations les plus enrichissantes pour le modèle**. Sous cet angle donc, le **modèle ALS obtient finalement de très bon score malgré cette pénalisation**, montrant ainsi une certaine **robustesse aux données**.  

Pour ce qui est de la manière de faire des recommandations avec ce modèle, la méthodologie est la suivante :
>* un nouvel utilisateur saisit ses films préférés
>* le système crée de nouveaux échantillons d'interaction utilisateur-film pour le modèle
>* le système ré-entraîne le modèle ALS sur les données avec les nouvelles entrées
>* le système crée des données de film pour la nouvelle entrée 
>* le système fait des prédictions de classement sur tous les films pour ce nouvel utilisateur
>* le système génère les N meilleures recommandations de films pour cet utilisateur en fonction du classement des prévisions de classement des films

## Conclusion

Dans ce projet nous avons vu comment améliorer un système de recommandation de filtrage collaboratif avec la factorisation matricielle. Nous avons appris que la factorisation matricielle peut **résoudre les problèmes de biais populaire et de démarrage à froid** dans le filtrage collaboratif KNN que nous avions initialement paramétré. Nous avons également utilisé Spark ML pour implémenter un système de recommandation distribué utilisant l'alternance des moindres carrés (ALS). 

>Une idée pour améliorer davantage notre système de recommandation de films serait finalement de **mélanger les listes de recommandation du système KNN** donnant des résultats peu "originaux" avec la liste obtenue **avec l'ALS** donnant des résultats plus "originaux". Cette implémentation hybride pourrait **offrir à la fois du contenu populaire et du contenu moins connu aux utilisateurs**.

