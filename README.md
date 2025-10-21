 Rumour Identification from Social Media Using Machine Learning Techniques

Identifying misinformation on social media by classifying user-generated content as rumour or non-rumour using NLP and ML algorithms.
## Table of Contents
- [Overview](#overview)
- [Abstract](#abstract)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Tools & Technologies](#tools--technologies)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Building & Training](#model-building--training)
- [Evaluation](#evaluation)
- [Web Application & User Interaction](#web-application--user-interaction)
- [Feedback Mechanism](#feedback-mechanism)
- [Key Findings](#key-findings)
- [How to Run This Project](#how-to-run-this-project)
- [Future Scope](#future-scope)
- [Author & Contact](#author--contact)

## <a id="overview"></a>Overview
With the rapid spread of unverified information online, social media platforms face a major challenge in identifying and controlling rumours.
This project proposes a machine learning-based solution to automatically detect and classify posts as *rumour* or *non-rumour*.  
A combination of classical and deep learning algorithms were tested, and a *Random Forest classifier
* achieved the highest accuracy of *93%*.
* A user-friendly web interface allows real-time prediction and user feedback, enabling continuous improvement of the model.
## <a id="abstract"></a>Abstract
The mass spread of misinformation on social media poses a serious challenge to public knowledge and online trustworthiness.  
This project presents a *machine learning-based rumour detection system* aimed at labelling user-generated content as *rumour* or *non-rumour. 
The system combines several classification algorithms — **Long Short-Term Memory (LSTM), **Random Forest, **Logistic Regression, **Naïve Bayes, and **BERT, all trained on the annotated **PHEME dataset*.  
Of these, the *Random Forest model proved to be most accurate at 93%.
To make it accessible, a simple **web application* was created so that authenticated users can enter statements and obtain instantaneous classification results.
The platform also includes a *feedback mechanism* allowing users to confirm or correct predictions, continuously improving model reliability.  
By integrating sophisticated NLP methods with an interactive interface, the system offers a scalable and efficient solution for real-time rumour detection in online communication networks.

---
## <a id="business-problem"></a>Business Problem
Unverified or misleading social media content can misinform the public and damage reputations.  
Organisations and platforms need an automated way to:
- Detect rumour content early.  
- Assess credibility of user posts.  
- Identify and monitor sources of false information.  
This system supports digital media monitoring and reputation management teams by providing automatic rumour classification.
---
## <a id="dataset"></a>Dataset
- *Dataset Used:* PHEME dataset (publicly available annotated dataset for rumour detection).  
- *Content:* Twitter conversations grouped around breaking news events.  
- *Labels:* Rumour and Non-Rumour.  
- *Size:* ~6,400 tweets and threads.  
- *Format:* CSV / JSON containing tweet text, user metadata, and thread structure.
---
## <a id="tools--technologies"></a>Tools & Technologies
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python 3.10 |
| Data Handling | Pandas, NumPy |
| NLP | NLTK, spaCy, Word2Vec, Transformers |
| ML Models | Scikit-learn (RF, LR, NB), TensorFlow / Keras (LSTM), Hugging Face (BERT) |
| Web Framework | Flask |
| Visualization | Matplotlib, Seaborn |
| Interface | HTML, CSS, Bootstrap |
| Version Control | Git, GitHub |
## <a id="project-structure"></a>Project Structure
## <a id="data-preprocessing"></a>Data Preprocessing
Steps performed:
1. Removed URLs, emojis, mentions, and punctuation.  
2. Lowercased and tokenised text.  
3. Removed stopwords using NLTK.  
4. Lemmatization for word normalisation.  
5. Feature extraction using *TF-IDF, *Word2Vec, and *BERT embeddings*.
---
   <a id="model-building--training"></a>Model Building & Training
Models implemented:
Logistic Regression – baseline linear classifier.
Naïve Bayes – probabilistic approach using word frequency.
Random Forest – ensemble of decision trees (highest accuracy: 93%).
LSTM – sequence-based deep learning for contextual text understanding.
BERT – fine-tuned transformer model for advanced rumour classification.
Each model was trained and validated using stratified k-fold cross-validation.

---
<a id="evaluation"></a>Evaluation

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.88	0.87	0.86	0.86
Naïve Bayes	0.85	0.84	0.82	0.83
Random Forest	0.93	0.92	0.93	0.93
LSTM	0.91	0.90	0.90	0.90
BERT	0.92	0.91	0.91	0.91
Visual evaluation: confusion matrix, ROC-AUC, and classification report.

---
<a id="web-application--user-interaction"></a>Web Application & User Interaction
The Flask-based web interface allows:
User login / registration (for authentication).
Text input box to submit a statement.
Instant rumour / non-rumour prediction from the deployed Random Forest model.
Prediction confidence score displayed to the user.

---
<a id="feedback-mechanism"></a>Feedback Mechanism
Each prediction page includes a simple feedback form where users can mark the prediction as:
 Correct
 Incorrect
These responses are stored in the database and later used to retrain and improve the model.
This ensures that the system learns continuously from real user interactions.

---
<a id="key-findings"></a>Key Findings
Random Forest performed best among classical algorithms.
LSTM and BERT captured deeper contextual semantics but required higher computation.
Ensemble of BERT + RF yielded stable and interpretable performance.
User feedback integration improved post-deployment accuracy by ~3%.

---
<a id="future-scope"></a>Future Scope
Integration with real-time social media APIs for live rumour tracking.
Add network graph visualisation to detect rumour propagation patterns.
Implement active learning to use user feedback for automated retraining.
Deploy as a cloud microservice using Docker and FastAPI.

---
<a id="author--contact"></a>Author & Contact

**Singavarapu sivani**
 Email: sivanisingavarapu@gmail.com
 GitHub: github.com/sivanisingavarapu
 Project: Rumour Identification from Social Media Using Machine Learning Algorithms

