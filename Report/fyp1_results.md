Please note that these preliminary results are not cross-validated. Cross-validation will be performed on all models during the second half of this final year project.

\subsubsection{GloVe Embedding}
Experimentation with GloVe embeddings showed that ML models trained using GloVe embeddings performed well, however, with the exception of ML models trained on the FakeNewsNet dataset. Results for the ML models range from 0.85 - 0.98 (F1-score) with FakeNewsNet excluded and an anomalous result from the Gaussian Naive Bayes model for the WELFake dataset excluded as well. Elaborating on the anomalous result, it shows that the Gaussian NB model for WELFake performing poorly relative to other ML models across all datasets, this seems to be an outlier which also appears in Word2Vec's results, indicating something about the dataset which is not particularly compatible with Gaussian Naive Bayes. 

For the DL models, the CNN models performed poorly on all datasets, with F1-scores in the range of 0.44 - 0.60 (excluding FakeNewsNet). Likely because CNN's are not optimal for sequential data, something which LSTMs are generally better at. For the CNN-LSTM models, it is clear that the hybridisation does not solve the drawbacks experienced by the CNN model with results between 0.61 - 0.68 (excluding FakeNewsNet). And lastly, LSTMs performed substantially better than its hybrid and CNN counterparts, with scores ranging from 0.93 - 0.98. This is likely because LSTMs are able to capture long-term dependencies and "remember" key discernible differences that CNNs are not able to.

Lastly, the unique performance of models on FakeNewsNet warrants additional discussion. For the ML models, while overall accuracy is high, this is compounded by the fact that the models actually perform very poorly on the negative class, with F1-scores for the negative class between 0.32 - 0.54 (results in Table \ref{tab:glove_ml} do not differentiate by class, this assessment was made observing the classification reports generated during experimentation). This trend can also be seen in the DL models, with the CNN and CNN-LSTM models failing to predict the negative class entirely (explaining their exact same results) and the LSTM model having an F1-score of 0.59 for the negative class, compared to 0.88 for the positive class (overall F1-score of 0.82). This can be attributed to the class imbalance of the dataset, which has three times the number of positive samples than negative samples. And potentially the content of the dataset as well, as the dataset only has the title feature. 

\begin{table}[H]
\centering
\small
\caption{Machine Learning Model Performance (GloVe Embeddings)}
\label{tab:glove_ml}
\begin{tabular}{llcccc}
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule
\textbf{WELFake} & Gaussian NB & 0.69 & 0.70 & 0.69 & 0.67 \\
 & KNN & 0.85 & 0.86 & 0.85 & 0.85 \\
 & Logistic Regression & 0.90 & 0.90 & 0.90 & 0.90 \\
 & Random Forest & 0.87 & 0.87 & 0.87 & 0.87 \\
 & SVM & 0.93 & 0.93 & 0.93 & 0.93 \\
 & XGBoost & 0.91 & 0.91 & 0.91 & 0.91 \\
\midrule
\textbf{ISOT} & Gaussian NB & 0.87 & 0.87 & 0.87 & 0.87 \\
 & KNN & 0.93 & 0.93 & 0.93 & 0.93 \\
 & Logistic Regression & 0.97 & 0.97 & 0.97 & 0.97 \\
 & Random Forest & 0.93 & 0.93 & 0.93 & 0.93 \\
 & SVM & 0.98 & 0.98 & 0.98 & 0.98 \\
 & XGBoost & 0.96 & 0.96 & 0.96 & 0.96 \\
\midrule
\textbf{Fake News Detection} & Gaussian NB & 0.87 & 0.87 & 0.87 & 0.87 \\
 & KNN & 0.93 & 0.93 & 0.93 & 0.93 \\
 & Logistic Regression & 0.97 & 0.97 & 0.97 & 0.97 \\
 & Random Forest & 0.94 & 0.94 & 0.94 & 0.93 \\
 & SVM & 0.98 & 0.98 & 0.98 & 0.98 \\
 & XGBoost & 0.96 & 0.96 & 0.96 & 0.96 \\
\midrule
\textbf{Fake News Classification} & Gaussian NB & 0.88 & 0.88 & 0.88 & 0.88 \\
 & KNN & 0.92 & 0.92 & 0.92 & 0.92 \\
 & Logistic Regression & 0.95 & 0.95 & 0.95 & 0.95 \\
 & Random Forest & 0.92 & 0.92 & 0.92 & 0.92 \\
 & SVM & 0.96 & 0.96 & 0.96 & 0.96 \\
 & XGBoost & 0.95 & 0.95 & 0.95 & 0.95 \\
\midrule
\textbf{FakeNewsNet} & Gaussian NB & 0.70 & 0.76 & 0.70 & 0.72 \\
 & KNN & 0.80 & 0.79 & 0.80 & 0.80 \\
 & Logistic Regression & 0.80 & 0.79 & 0.80 & 0.79 \\
 & Random Forest & 0.79 & 0.76 & 0.79 & 0.74 \\
 & SVM & 0.83 & 0.81 & 0.83 & 0.81 \\
 & XGBoost & 0.81 & 0.79 & 0.81 & 0.79 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\small
\caption{Deep Learning Model Performance (GloVe Embeddings)}
\label{tab:glove_dl}
\begin{tabular}{llcccc}
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule
\textbf{WELFake} & CNN & 0.5690 & 0.69 & 0.57 & 0.44 \\
 & CNN-LSTM & 0.6344 & 0.65 & 0.63 & 0.61 \\
 & LSTM & 0.9310 & 0.93 & 0.93 & 0.93 \\
\midrule
\textbf{ISOT} & CNN & 0.5887 & 0.64 & 0.59 & 0.51 \\
 & CNN-LSTM & 0.6758 & 0.68 & 0.68 & 0.67 \\
 & LSTM & 0.9763 & 0.98 & 0.98 & 0.98 \\
\midrule
\textbf{Fake News Detection} & CNN & 0.6292 & 0.64 & 0.63 & 0.60 \\
 & CNN-LSTM & 0.6561 & 0.71 & 0.66 & 0.62 \\
 & LSTM & 0.9770 & 0.98 & 0.98 & 0.98 \\
\midrule
\textbf{Fake News Classification} & CNN & 0.5920 & 0.63 & 0.59 & 0.52 \\
 & CNN-LSTM & 0.6804 & 0.68 & 0.68 & 0.68 \\
 & LSTM & 0.9614 & 0.96 & 0.96 & 0.96 \\
\midrule
\textbf{FakeNewsNet} & CNN & 0.7645 & 0.58 & 0.76 & 0.66 \\
 & CNN-LSTM & 0.7645 & 0.58 & 0.76 & 0.66 \\
 & LSTM & 0.8194 & 0.81 & 0.82 & 0.82 \\
\bottomrule
\end{tabular}
\end{table}


\subsubsection{Word2Vec}
The results for the models trained using the Word2Vec embeddings showed interest results, especially pertaining to the datasets used and its performance on different models. And if comparing to GloVe embedding results, it seems that the results for Word2Vec performed on-par if not slighly better in some cases.

Firstly, the machine learning models generally did very well on all datasets save for FakeNewsNet, indicating the dataset is more challenging than its peers. Notably the Gaussian Naive Bayes model performed relatively poorly on the WELFake dataset with an F1-score of 0.69 while other ML models (across all datasets) had F1-scores at or exceeding 0.87 (except for FakeNewsNet), going as high as 0.99. This seems to be an outlier which also appears in the GloVe embedding results, the cause of which was elaborated on in the GloVe embedding results discussion section.

As for the DL approaches, the results indicate that the CNN's did not perform well with F1-scores betweem 0.43 and 0.68 (FakeNewsNet excluded). This may be because CNNs are not usually preferred for sequential data. And while CNN-LSTMs performed better than LSTMs (except for FakeNewsNet) with F1-scores between 0.65 and 0.70 (FakeNewsNet excluded), it seems the CNN component is holding the model back as LSTMs performed significantly better than the CNN-LSTM hybrid. The final DL approach employed were LSTMs, which performed by far the best with F1-scores between 0.95 and 0.99 (excluding FakeNewsNet). This is likely because LSTMs are less vulnerable to the vanishing gradient problem and are able to remember long-term depencies in the data. 

Finally, it is worth discussing the results for FakeNewsNet, which didn't really follow the trends in the other four datasets. FakeNewsNet, which as previously mentioned seems to be a more challenging dataset, saw high accuracies by the ML models but when this is broken down into the positive and negative class, it is clear that the models excel at detecting the positive class possibly due to the class imbalance. This trend can also be applied to the DL models and their results, which saw the CNN and CNN-LSTM models completely fail to classify the negative class and instead just predict the positive class likely due to the imbalance (this can be seen in their identical results). Only the LSTM model was able to predict the negative class but this was limited to an F1-score of 0.60. It could be argued that the FakeNewsNet dataset is challenging not because of its content but because of its class imbalance. The impact of this class imbalance could be reduced by modifying the training configuration for the DL models.

\begin{table}[H]
\centering
\small
\caption{Machine Learning Model Performance (Word2Vec Embeddings)}
\label{tab:word2vec_ml}
\begin{tabular}{llcccc}
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule
\textbf{WELFake} & Gaussian NB & 0.71 & 0.73 & 0.71 & 0.69 \\
 & KNN & 0.87 & 0.88 & 0.87 & 0.87 \\
 & Logistic Regression & 0.93 & 0.93 & 0.93 & 0.93 \\
 & Random Forest & 0.90 & 0.90 & 0.90 & 0.90 \\
 & SVM & 0.95 & 0.95 & 0.95 & 0.95 \\
 & XGBoost & 0.93 & 0.93 & 0.93 & 0.93 \\
\midrule
\textbf{ISOT} & Gaussian NB & 0.91 & 0.91 & 0.91 & 0.91 \\
 & KNN & 0.95 & 0.95 & 0.95 & 0.95 \\
 & Logistic Regression & 0.98 & 0.98 & 0.98 & 0.98 \\
 & Random Forest & 0.95 & 0.95 & 0.95 & 0.95 \\
 & SVM & 0.99 & 0.99 & 0.99 & 0.99 \\
 & XGBoost & 0.97 & 0.97 & 0.97 & 0.97 \\
\midrule
\textbf{Fake News Detection} & Gaussian NB & 0.90 & 0.90 & 0.90 & 0.90 \\
 & KNN & 0.95 & 0.95 & 0.95 & 0.95 \\
 & Logistic Regression & 0.98 & 0.98 & 0.98 & 0.98 \\
 & Random Forest & 0.96 & 0.96 & 0.96 & 0.96 \\
 & SVM & 0.99 & 0.99 & 0.99 & 0.99 \\
 & XGBoost & 0.98 & 0.98 & 0.98 & 0.98 \\
\midrule
\textbf{Fake News Classification} & Gaussian NB & 0.90 & 0.90 & 0.90 & 0.90 \\
 & KNN & 0.93 & 0.93 & 0.93 & 0.93 \\
 & Logistic Regression & 0.96 & 0.96 & 0.96 & 0.96 \\
 & Random Forest & 0.94 & 0.94 & 0.94 & 0.94 \\
 & SVM & 0.97 & 0.97 & 0.97 & 0.97 \\
 & XGBoost & 0.96 & 0.96 & 0.96 & 0.96 \\
\midrule
\textbf{FakeNewsNet} & Gaussian NB & 0.76 & 0.78 & 0.76 & 0.76 \\
 & KNN & 0.81 & 0.80 & 0.81 & 0.80 \\
 & Logistic Regression & 0.83 & 0.81 & 0.83 & 0.81 \\
 & Random Forest & 0.82 & 0.80 & 0.82 & 0.79 \\
 & SVM & 0.83 & 0.82 & 0.83 & 0.81 \\
 & XGBoost & 0.82 & 0.81 & 0.82 & 0.81 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\small
\caption{Deep Learning Model Performance (Word2Vec Embeddings)}
\label{tab:word2vec_dl}
\begin{tabular}{llcccc}
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule
\textbf{WELFake} & CNN & 0.5673 & 0.71 & 0.57 & 0.43 \\
 & CNN-LSTM & 0.6457 & 0.65 & 0.65 & 0.65 \\
 & LSTM & 0.9540 & 0.95 & 0.95 & 0.95 \\
\midrule
\textbf{ISOT} & CNN & 0.6726 & 0.68 & 0.67 & 0.66 \\
 & CNN-LSTM & 0.7005 & 0.70 & 0.70 & 0.70 \\
 & LSTM & 0.9875 & 0.99 & 0.99 & 0.99 \\
\midrule
\textbf{Fake News Detection} & CNN & 0.6860 & 0.68 & 0.69 & 0.68 \\
 & CNN-LSTM & 0.6859 & 0.69 & 0.69 & 0.68 \\
 & LSTM & 0.9893 & 0.99 & 0.99 & 0.99 \\
\midrule
\textbf{Fake News Classification} & CNN & 0.6508 & 0.66 & 0.65 & 0.63 \\
 & CNN-LSTM & 0.6827 & 0.69 & 0.68 & 0.67 \\
 & LSTM & 0.9708 & 0.97 & 0.97 & 0.97 \\
\midrule
\textbf{FakeNewsNet} & CNN & 0.7645 & 0.58 & 0.76 & 0.66 \\
 & CNN-LSTM & 0.7645 & 0.58 & 0.76 & 0.66 \\
 & LSTM & 0.8260 & 0.82 & 0.83 & 0.82 \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{BERT Fine-tuning}
The results from the baseline fine-tuned BERT models exhibited very high accuracy, recall and precision. As can be referred to in Table \ref{tab:bert_test}, it appears that the FakeNewsNet dataset was the most challenging dataset. All the other datasets proved to be no challenge for each of their fine-tuned BERT models. It is however notable that the F1-scores are in the range (with the exception of the FakeNewsNet dataset) exceed 0.98, this may indicate that these datasets may have gathered the data for their positive and negative classes with a methodology that unknowingly introduced some level of data leakage. The alternative is that BERT is simply a very effective model, but the relatively lower performance of BERT on the FakeNewsNet dataset indicates the other datasets may be of lower quality or are less challenging. 

Referencing the results from GloVe and Word2Vec, it is clear that the FakeNewsNet dataset is challenging due to its class imbalance and potentially due to the fact that FakeNewsNet only contains the titles of the articles, not the rest of the content. 

BERT has either matched or outperformed the already excellent results obtained from the machine learning models and deep learning models trained on GloVe and Word2Vec vector embeddings. This includes FakeNewsNet where it was able to obtain an F1-score of 0.9016, which was not able to be reached by any of the other machine learning or deep learning models trained on GloVe or Word2Vec embeddings. 

\begin{table}[H]
\centering
\caption{BERT Fine-tuning Results (Evaluated using test set)}
\label{tab:bert_test}
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule
WELFake & 0.9898 & 0.9844 & 0.9933 & 0.9889 \\
FakeNewsNet & 0.8499 & 0.8934 & 0.91 & 0.9016 \\
Fake News Detection & 0.9969 & 0.9950 & 0.9994 & 0.9972 \\
ISOT & 0.9997 & 0.9993 & 1.00 & 0.9996 \\
Fake News Classification & 0.9974 & 0.9975 & 0.9978 & 0.9976 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Gantt chart for FYP2}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.5\linewidth]{FYP2-Gantt_Chart.png}
    \caption{Figure depicts planned timeline for FYP2 for MMU Term 2610}
    \label{fig:FYP2-GANTT_CHART}
\end{figure}
