# PySpark-Movie-Recommender

For any given user we would like to use their movie ratings, in combination with all the existing user ratings, to determine which movies they might prefer. For example, a user might highly rate Annie Hall and The Purple Rose of Cairo (both Woody Allen movies that our database does not have information for), can we infer from other users that they might like Zelig (another Woody Allen movie)? These might also include affinities for an actor, or director, or genre, etc.
 
## Motivation

- Familiarize myself with the [Apache Spark](https://spark.apache.org) via [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) for big data applications
- Familiarize myself with training and tuning an Alternating Least Squares (ALS) algorithm for model-based movie recommendations
- [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize)

## Input

User movie ratings:

```python
ratings_raw_RDD = sc.textFile('data/ratings.csv')
ratings_RDD = ratings_raw_RDD.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2])))
ratings_RDD.take(3) # [(user_id, movie_id, rating)]
[(1, 31, 2.5), (1, 1029, 3.0), (1, 1061, 3.0)]
```

New user movie ratings: 
```python 
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
```

Movie lookup (to map `movie_id` to `movie_title`): 
```python 
movies_raw_RDD = sc.textFile('data/movies.csv')
movies_RDD = movies_raw_RDD.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1]))
movies_RDD.take(3)
[(1, u'Toy Story (1995)'), (2, u'Jumanji (1995)'), (3, u'Grumpier Old Men (1995)')] 
```

## Output
Top recommendations for new user with predicted ratings: 

```python
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
```

Notice that out top ranked movies have predicted ratings higher than 5. This makes sense as there is no ceiling implied in our algorithm and one can imagine that certain combinations of factors would combine to create “better than anything you’ve seen yet” ratings.

Nevertheless, we constrain our ratings to a scaled range of 1-5 via [`MinMaxScaler`](https://spark.apache.org/docs/2.1.0/ml-features.html#minmaxscaler). 

## References
- [PSC workshops](https://www.psc.edu/current-workshop)
- [PySpark](https://spark.apache.org/docs/2.1.0/ml-features.html)
- Varun Abhi's [Medium article](https://medium.com/@varunabhi86/movie-recommendation-using-apache-spark-1a41e24b94ba) in Spark


## License
[MIT](https://choosealicense.com/licenses/mit/)
