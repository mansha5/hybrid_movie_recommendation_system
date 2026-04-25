
Hybrid Movie Recommendation System

---

##OVERVIEW

>>>>>>> b651443 (Improve README with clean structure and better documentation)
This project demonstrates an end-to-end hybrid recommendation system combining content-based and collaborative filtering techniques, evaluated using Precision@K.

---

##ABOUT THE DATASET

MovieLens Latest Datasets
Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
Link: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
Files used: movies.csv(movieId,title,genres), ratings.csv(userId,movieId,rating,timestamp)

---

##PREREQUISITES

-Git
-Python 3.13.5
-VSCode

---

##ARCHITECTURE
1. Content-Based Filtering
-Uses movie genres + titles
-Converts text → TF-IDF vectors
-Computes similarity using cosine similarity

2. Collaborative Filtering
-Uses user rating patterns
-Finds similar users / preferences

3. Hybrid Model
-Combines both approaches:
Hybrid Score = α × Collaborative + (1 - α) × Content

---

##PROJECT STRUCTURE

project/ 
│ 
├── data/ 
│  ├── movies.csv 
│  ├── ratings.csv 
│ 
├── src/ 
│   ├── preprocess.py 
│   ├── content_based.py 
│   ├── collaborative.py 
│   ├── hybrid.py 
│   
├── main.py # Run system 
├── requirements.txt 
└── README.md

---

##INSTALLATION

1. First and foremost clone the repository onto your system to run on your desired IDE
Command- git clone <https://github.com/mansha5/hybrid_movie_recommendation_system.git> 

2. Change the directory to your project 

    ```
    cd project 
    ```

3. Create a virtual environment so that python can run smoothly

    ```
    python3 -m venv venv 
    ```

4. Activate your virtual environment 

    ```
    source venv/bin/activate 
    ```

5. Download the required libraries 

    ```
    pip install -r requirements.txt
    ```

6. Run the Project

    ```
    python main.py
    ````

---

##EVALUATION
Metric: Precision@10
Current Score: ~0.002

Why low?
-Limited features (only genres + titles)
-Sparse dataset (MovieLens small)
-Strict evaluation setup
-The system is implemented as a learning project to demonstrate recommendation system architecture.

---

##LIMITATIONS
-Cold start problem
-Limited feature representation
-Low precision due to dataset constraints

---

##FUTURE IMPROVEMENTS
-Add tags.csv for better features
-Use NLP embeddings (BERT / Sentence Transformers)
-Improve hybrid weighting

---

