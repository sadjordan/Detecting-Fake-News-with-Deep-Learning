# Results
However, even if the issues mentioned above were rectified, it is unlikely to produce a practical generalised model. Real world data and the datasets available continue to have substantial differences. This is exacerbated by the impossibility of knowing the ground truth of all the data in the wild. Nonetheless, this research has proven that current approaches to fake news detection are more than capable of performing well when working with a dataset. However, the actual process of generalising a model to a degree where it can be reliably in real-world settings remains elusive. Although there are likely more successful attempts than the one described here like by \textcite{BERT-finetuning}, there remains significant work to be done in regards to generalisation. 

# Conclusion
Additionally this research suffered from computational resource limitations. This impacted decisions related to batch size, epochs and token cap and contributed to the decision not to conduct hyperparameter optimisation. 

In concluion, this research has demonstrated that while BERT can perform very well on the datasets it is trained on, BERT doesn't generalise very well. Furthermore generalisation performance likely depends on similarity between the dataset used to train a model and the dataset being tested. And lastly, he experiments with the proposed model, TPA-BERT, showed that adversarial training on a perturbed dataset failed to increase generalisation compared to BERT. 



