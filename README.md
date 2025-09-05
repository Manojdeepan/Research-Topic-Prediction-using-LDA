# Research Topic Prediction using LDA (Course Recommendation System)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)  
![NLP](https://img.shields.io/badge/NLP-TextProcessing-green.svg)  
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## Project Overview
This repository contains a **Natural Language Processing (NLP)** project that predicts research topics and recommends similar courses using **Latent Dirichlet Allocation (LDA)**-style topic modeling and **cosine similarity**.  
The dataset used is **Udemy courses data** (`udemy_courses.csv`), and the system identifies related topics from course titles to recommend similar learning paths.

## Key Features
- **Text Cleaning:** Removes stopwords and special characters using `neattext`.
- **Vectorization:** Converts course titles into numerical format using **CountVectorizer**.
- **Cosine Similarity:** Computes similarity scores between course titles.
- **Recommendation Engine:** Suggests similar courses based on input course title.
- **Exploratory Data Analysis (EDA):** Inspects dataset, explores text features, and creates sparse/dense matrices.

## Repository Contents
- `Research_Topic_Prediction.ipynb` – Jupyter Notebook with full implementation.
- `udemy_courses.csv` – Dataset (must be uploaded in Colab or project folder).
- `README.md` – Project documentation.

## Installation

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn neattext
   ```
3. Download the Udemy courses dataset (`udemy_courses.csv`) and place it in your working directory.

## Usage

### For Jupyter Notebook:
1. Open `Research_Topic_Prediction.ipynb` in Jupyter Notebook or Google Colab
2. Upload the dataset when prompted
3. Run all cells to execute the entire pipeline

### For custom recommendations:
```python
# Call the recommendation function with your desired course title
recommend_course('Options Trading - How to Win with Weekly Options', 10)
```

## Example Input & Output

### Input Course:
```
"Options Trading - How to Win with Weekly Options"
```

### Output Recommendations:
```
Recommended Courses:
------------------------------------------------
1. How To Maximize Your Profits Trading Options (score: 0.89)  
2. Stock Market for Beginners - Learn to Trade from Scratch (score: 0.74)  
3. Advanced Options Trading Strategies (score: 0.72)  
4. Day Trading with Technical Analysis (score: 0.69)  
...
```

## Dataset Format
The expected dataset format (`udemy_courses.csv`):
```csv
course_id,course_title,price,num_subscribers,...
1,"How To Maximize Your Profits Trading Options",50,2000,...
2,"Options Trading - How to Win with Weekly Options",40,1500,...
3,"Python for Data Science and Machine Learning Bootcamp",30,3000,...
```

## Project Workflow
1. **Data Loading & Exploration**: Load the dataset and perform initial analysis
2. **Text Preprocessing**: Clean course titles using neattext library
3. **Feature Engineering**: Convert text to numerical features using CountVectorizer
4. **Similarity Calculation**: Compute cosine similarity between course vectors
5. **Recommendation Generation**: Generate course recommendations based on similarity scores

## Tools and Technologies
- Python 3.x
- pandas – Data handling and manipulation
- neattext – Text preprocessing (stopword and special character removal)
- scikit-learn – CountVectorizer & cosine similarity
- Jupyter Notebook / Google Colab – Execution environment

## Author
Manoj Deepan M

## License
This project is open source and available under the [MIT License](LICENSE).
