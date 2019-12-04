# coding: utf-8

'''
recommender.py 

Utilizing PySpark's Alternating Least Squares (ALS) algorithm for 
model-based movie recommendations. 

Author: Shan Sabri 
Date:   December 4, 2019

'''

from pyspark.mllib.recommendation import ALS
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
import math


############################################################################################################

### HYPERPARAMETERS
SEED = 5             # reproducable matrix approximation 
ITERATIONS = 10      # sufficiently large for convergence, we should experimentally benchmark this hyperparam
REGULARIZATION = 0.1 # lambda that penalizes large terms
RANKS = [4, 8, 12]   # num rows/ranks 
ERRORS = [0, 0, 0]   # placeholder for error of ranks
ERR = 0              # init error 

min_error = float('inf') # init
best_rank = -1           # init
best_iteration = -1      # init

############################################################################################################


### DATASET 
ratings_raw_RDD = sc.textFile('data/ratings.csv')
# ratings_raw_RDD = sc.textFile('data/ratings-large.csv') # competition size (larger rank [k=12] is better)


### PARSE [(user_id, movie_id, rating)]
ratings_RDD = ratings_raw_RDD.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2])))
# ratings_RDD.take(10)
# [(1, 31, 2.5), (1, 1029, 3.0), (1, 1061, 3.0), (1, 1129, 2.0), (1, 1172, 4.0), 
# (1, 1263, 2.0), (1, 1287, 2.0), (1, 1293, 2.0), (1, 1339, 3.5), (1, 1343, 2.0)]


### PARTITION INTO TRAINING (3/5), VALIDATION (1/5) and TEST SETS (1/5)
training_RDD, validation_RDD, test_RDD = ratings_RDD.randomSplit([3, 1, 1], 0) 


### CREATE PREDICTION SETS WITHOUT RATINGS 
predict_validation_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
predict_test_RDD = test_RDD.map(lambda x: (x[0], x[1]))

for rank in RANKS:

    # Alternating Least Squares (ALS) algorithm
    model = ALS.train(training_RDD, rank, seed = SEED, iterations = ITERATIONS, lambda_ = REGULARIZATION)
    
    # Coercing ((u,p),r) tuple format to accomodate join
    predictions_RDD = model.predictAll(predict_validation_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    # model.predictAll(predict_validation_RDD).take(2)
    # [Rating(user=463, product=4844, rating=2.7640960482284322), Rating(user=380, product=4844, rating=2.399938320644199)]

    ratings_and_preds_RDD = validation_RDD.map(lambda r: ((r[0], r[1]), r[2])).join(predictions_RDD) # join and look at common elements
    # ratings_and_preds_RDD.take(2)
    # [((119, 145), (4.0, 2.903215714486778)), ((407, 5995), (4.5, 4.604779028840272))]

    # Compute min error and best rank
    error = math.sqrt(ratings_and_preds_RDD.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    ERRORS[ERR] = error
    ERR += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

# For rank 4 the RMSE is 0.945874851075
# For rank 8 the RMSE is 0.950209172802
# For rank 12 the RMSE is 0.949442071644

print 'The best model was trained with rank %s' % best_rank
# The best model was trained with rank 4


############################################################################################################


### TESTING 
# Redo the last phase with the best rank size and using test dataset this time
model = ALS.train(training_RDD, best_rank, seed = SEED, iterations = ITERATIONS, lambda_ = REGULARIZATION)
predictions_RDD = model.predictAll(predict_test_RDD).map(lambda r: ((r[0], r[1]), r[2]))
ratings_and_preds_RDD = test_RDD.map(lambda r: ((r[0], r[1]), r[2])).join(predictions_RDD)
error = math.sqrt(ratings_and_preds_RDD.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print 'For testing data the RMSE is %s' % (error)
# For testing data the RMSE is 0.94100201562


############################################################################################################


### ADD NEW USER WITH 10 DATA POINTS
new_user_ID = 0 # We don't want to overwrite an existing user. 
                # ID of 0 is unused; ratings_RDD.filter(lambda x: x[0]=='0').count()
new_user = [
     (0,100,4), # City Hall (1996)
     (0,237,1), # Forget Paris (1995)
     (0,44,4),  # Mortal Kombat (1995)
     (0,25,5),  # etc....
     (0,456,3),
     (0,849,3),
     (0,778,2),
     (0,909,3),
     (0,478,5),
     (0,248,4)
    ]
new_user_RDD = sc.parallelize(new_user)
updated_ratings_RDD = ratings_RDD.union(new_user_RDD) # We are joining, and then training, with ALL data now - the ratings RDD. 

# UPDATE MODEL
updated_model = ALS.train(updated_ratings_RDD, best_rank, seed = SEED, iterations = ITERATIONS, lambda_ = REGULARIZATION)


############################################################################################################


### USE MOVIES DATASET TO GET MOVIE NAMES VIA IDX
# Use large or small datasets
movies_raw_RDD = sc.textFile('data/movies.csv')
# movies_raw_RDD = sc.textFile('data/movies-large.csv')

# Parse lines to get movie names 
movies_RDD = movies_raw_RDD.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1]))
# movies_RDD.take(5)
# [(1, u'Toy Story (1995)'), (2, u'Jumanji (1995)'), (3, u'Grumpier Old Men (1995)'), 
# (4, u'Waiting to Exhale (1995)'), (5, u'Father of the Bride Part II (1995)')]


# Create prediction type RDD of all movies not yet rated by new user
new_user_rated_movie_ids = map(lambda x: x[1], new_user)
# filter out all moves that are not in the short list of the new user
new_user_unrated_movies_RDD = movies_RDD.filter(lambda x: x[0] not in new_user_rated_movie_ids).map(lambda x: (new_user_ID, x[0])) 
# new_user_unrated_movies_RDD.take(3)
# [(0, 1), (0, 2), (0, 3)]

# GET RECOMENDATIONS
new_user_recommendations_RDD = updated_model.predictAll(new_user_unrated_movies_RDD) # will inclued nearly all movies 
# new_user_recommendations_RDD.take(2)
# [Rating(user=0, product=4704, rating=3.606560950463134), Rating(user=0, product=4844, rating=2.1368358868224036)] 

print "There are %s recommendations in the complete dataset" % (new_user_recommendations_RDD.count())
# There are 9057 recommendations in the complete dataset



# TRANSOFRM INTO STRUCT [(movie, predicted_rating)]
product_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))

# Join with Movies to get real title and format 
new_user_recommendations_titled_RDD = product_rating_RDD.join(movies_RDD)
new_user_recommendations_formatted_RDD = new_user_recommendations_titled_RDD.map(lambda x: (x[1][1],x[1][0]))
# new_user_recommendations_formatted_RDD.take(5)
# [(u'Beyond the Valley of the Dolls (1970)', 1.8833597779874536), (u'Heat (1995)', 2.4494414977308594),
# (u'Dracula: Dead and Loving It (1995)', 1.5132972218190268), (u'"Razor\'s Edge', 2.1754132876101195), 
# (u'Four Rooms (1995)', 3.286666746567237)]


############################################################################################################


### TOP 10 RECOMMENDATIONS
top_recomends = new_user_recommendations_formatted_RDD.takeOrdered(10, key=lambda x: -x[1])
for line in top_recomends: 
    print line

# (u'Hear My Song (1991)', 6.768762414140875)
# (u'Novocaine (2001)', 6.082847646559083)
# (u'Let It Be (1970)', 5.960487606934326)
# (u'"Broken Hearts Club', 5.607092763004114)
# (u'Evangelion: 1.0 You Are (Not) Alone (Evangerion shin gekij\xf4ban: Jo) (2007)', 5.519627831466998)
# (u'Six-String Samurai (1998)', 5.486386537669752)
# (u'"Play House', 5.46740481439733)
# (u'"Goat', 5.46740481439733)
# (u'Land of Silence and Darkness (Land des Schweigens und der Dunkelheit) (1971)', 5.46740481439733)
# (u'Cops (1922)', 5.46740481439733)

# We noticed that out top ranked movies have ratings higher than 5. This makes  sense as there is no ceiling 
# implied in our algorithm and one can imagine that certain combinations of factors would combine to create 
# “better than anything you’ve seen yet” ratings.
# Nevertheless, we may have to constrain our ratings to a 1-5 range. 



### SCALE PREDICTED RATINGS WITHIN DEFINED BOUNDS  
new_user_recommendations_formatted_RDD_DF = new_user_recommendations_formatted_RDD.toDF(['movie', "rating"])
to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
new_user_recommendations_formatted_RDD_DF = new_user_recommendations_formatted_RDD_DF.select("movie", to_vector("rating").alias("rating"))
scaler = MinMaxScaler(inputCol="rating", outputCol="scaled_rating", min = 1, max = 5)
model = scaler.fit(new_user_recommendations_formatted_RDD_DF)
new_user_recommendations_formatted_RDD_DF_scaled = model.transform(new_user_recommendations_formatted_RDD_DF)
print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
# Features scaled to range: [1.000000, 5.000000]

new_user_recommendations_formatted_RDD_DF_scaled.select("rating", "scaled_rating").show()
# +--------------------+--------------------+
# |              rating|       scaled_rating|
# +--------------------+--------------------+
# |[1.8833597779874536]| [2.810434087306585]|
# |[2.4494414977308594]|[3.0641436235844264]|
# |[1.5132972218190268]|[2.6445774693577806]|
# |[2.1754132876101195]| [2.941328193069741]|
# | [3.286666746567237]|[3.4393757185877063]|
# |[2.9038808064793997]|[3.2678166663056722]|
# |  [4.62265956524149]| [4.038148133725953]|
# | [3.240167362165092]|[3.4185353755022225]|
# |[2.0167157748171416]|[2.8702022920295103]|
# | [2.842788027979031]| [3.240435777711921]|
# |[1.3239490794775017]|[2.5597144050455922]|
# |[1.8968476376230612]|[2.8164791484597376]|
# | [2.371490825998972]|[3.0292072741354317]|
# | [2.446899937943037]|[3.0630045337097647]|
# | [2.967482885089602]|[3.2963221864588674]|
# |[1.6127366684869422]|[2.6891447730207174]|
# |  [3.30259073897302]|[3.4465126187702353]|
# |[1.4593965568820657]|[2.6204199807315405]|
# |[1.3239490794775017]|[2.5597144050455922]|
# |[0.06887811577384...| [1.997209980169669]|
# +--------------------+--------------------+
# only showing top 20 rows



#Top 10 recommedations by scaled ratings value 
top_recomends = new_user_recommendations_formatted_RDD_DF_scaled.rdd.takeOrdered(10, key=lambda x: -x[2])
new_user_recommendations_formatted_RDD_DF_scaled.orderBy('scaled_rating', ascending=False).show()
# +--------------------+--------------------+--------------------+
# |               movie|              rating|       scaled_rating|
# +--------------------+--------------------+--------------------+
# | Hear My Song (1991)| [6.768762414140875]|               [5.0]|
# |    Novocaine (2001)| [6.082847646559083]| [4.692583046687709]|
# |    Let It Be (1970)| [5.960487606934326]| [4.637743068943494]|
# | "Broken Hearts Club| [5.607092763004114]| [4.479356675514017]|
# |Evangelion: 1.0 Y...| [5.519627831466998]|[4.4401561741607605]|
# |Six-String Samura...| [5.486386537669752]| [4.425257913113933]|
# |         Cops (1922)|  [5.46740481439733]| [4.416750582738764]|
# |               "Goat|  [5.46740481439733]| [4.416750582738764]|
# |Land of Silence a...|  [5.46740481439733]| [4.416750582738764]|
# |         "Play House|  [5.46740481439733]| [4.416750582738764]|
# |  Dersu Uzala (1975)| [5.441536583883281]| [4.405156820673771]|
# |                "Now|[5.4369177354878655]| [4.403086720467973]|
# |    The Witch (2015)|  [5.37433033977239]| [4.375035966327726]|
# |              "Norte|  [5.35937346933437]|[4.3683325160472295]|
# |"Secret in Their ...| [5.358001407621149]|   [4.3677175780818]|
# |Book of Shadows: ...| [5.354853467836861]| [4.366306717573419]|
# |   Angel Baby (1995)| [5.351015314453442]| [4.364586513438384]|
# |       Gabbeh (1996)| [5.351015314453442]| [4.364586513438384]|
# |Picture Bride (Bi...| [5.351015314453442]| [4.364586513438384]|
# |King Kong vs. God...|[5.3293427074850825]|  [4.35487316839985]|
# +--------------------+--------------------+--------------------+
# only showing top 20 rows


## DONE! 