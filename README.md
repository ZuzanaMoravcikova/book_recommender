# Book Recommender Project

This project explores several approaches to book recommendation using two datasets and a shared notebook-based workflow.

## Included methods

- **User-based KNN**  
  Collaborative filtering based on similarity between users and their rating patterns.

- **Biased Matrix Factorization**  
  A latent-factor model for explicit ratings that captures hidden user and item preferences.

- **Hybrid Item-to-Item KNN**  
  A combination of collaborative similarity from ratings and content similarity from book metadata.

- **TF-IDF Content-Based Recommender**  
  A metadata-driven approach that recommends books with similar title, author, genre, and description.

## Notebooks

- `kaggle_recommenders.ipynb`  
  Main notebook with collaborative and hybrid recommenders.

- `goodreads_recommender.ipynb`  
  Content-based recommender built on richer Goodreads text metadata.