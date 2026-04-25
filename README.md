
Hybrid Movie Recommendation System

---

##OVERVIEW


This project demonstrates an end-to-end hybrid recommendation system combining content-based and collaborative filtering techniques, evaluated using Precision@K.

---

##ABOUT THE DATASET

MovieLens Latest Datasets
Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018.
Link: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
Files used: movies.csv(movieId,title,genres), ratings.csv(userId,movieId,rating,timestamp)

---

##PREREQUISITES

-Git <br>
-Python 3.13.5 <br>
-VSCode

---

##ARCHITECTURE
1. Content-Based Filtering <br>
-Uses movie genres + titles <br>
-Converts text → TF-IDF vectors <br>
-Computes similarity using cosine similarity

2. Collaborative Filtering <br>
-Uses user rating patterns <br>
-Finds similar users / preferences 

3. Hybrid Model <br>
-Combines both approaches: <br>
Hybrid Score = α × Collaborative + (1 - α) × Content

---

##PROJECT STRUCTURE

project/ <br>
│<br>
├── data/ <br>
│  ├── movies.csv <br>
│  ├── ratings.csv <br>
│ <br>
├── src/ <br>
│   ├── preprocess.py <br>
│   ├── content_based.py <br>
│   ├── collaborative.py <br>
│   ├── hybrid.py <br>
│   <br>
├── main.py # Run system <br>
├── requirements.txt <br>
└── README.md<br>

---

##INSTALLATION

1. First and foremost clone the repository onto your system to run on your desired IDE

    ```
   git clone <https://github.com/mansha5/hybrid_movie_recommendation_system.git>
    ```

3. Change the directory to your project 

    ```
    cd project 
    ```

4. Create a virtual environment so that python can run smoothly

    ```
    python3 -m venv venv 
    ```

5. Activate your virtual environment 

    ```
    source venv/bin/activate 
    ```

6. Download the required libraries 

    ```
    pip install -r requirements.txt
    ```

7. Run the Project

    ```
    python main.py
    ````

---

##EVALUATION <br>
Metric: Precision@10 <br>
Current Score: ~0.002 <br>

Why low? <br>
-Limited features (only genres + titles)<br>
-Sparse dataset (MovieLens small)<br>
-Strict evaluation setup<br>
-The system is implemented as a learning project to demonstrate recommendation system architecture.

---

##LIMITATIONS<br>
-Cold start problem<br>
-Limited feature representation<br>
-Low precision due to dataset constraints

---

##FUTURE IMPROVEMENTS<br>
-Add tags.csv for better features<br>
-Use NLP embeddings (BERT / Sentence Transformers)<br>
-Improve hybrid weighting

---

##AUTHOR <br>

Mansha Malhotra<br>






