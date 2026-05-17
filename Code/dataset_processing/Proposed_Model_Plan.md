Please formulate a plan to construct a new notebook in order to develop my proposed model. I will likely iterate on this plan multiple times.

My proposed model:
- A BERT based model
- Requires fine tuning/adversarial training to the BERT transformer in order to modify the embedding output.
    - For the adversarial training part, the idea is that we will perturb the text in the dataset and then pass both the original and perturbed text into the BERT model, and using a custom loss functions, penalise differences between the two output embeddings.
    - Let's do two experiments regarding the output embeddings: One using the current sequential vector output with variable length, and another using averaged embeddings fixed to a certain length.
    - Also attempt another experiment: I would like to create a hybrid between the variable length and averaged vector techniques. In order to do this, the flow will be to split the text into chunks of a set number of tokens (I am unsure how many tokens are in each text, so please for now use 128 tokens per chunk).To prevent premature cut off, ensure there is chunk overlap. In each chunk, the embeddings will be averaged, similar to a sliding window approach.
    - Please also explore if hyperparameter tuning is possible, and if it can be feasibly implemented with limited computational resources.
    - In order to ensure consistency, we will be using the same training parameters as the baseline notebook 'New_FYP_7_1_3.ipynb'. 
- The idea here is to generalise the model so that it performs better when tested against completely unseen datasets.
- Right now I've focused a lot on the BERT transformer training. But regarding the the classifier, I am unsure if it would still be possible to use the BERT classifier to do the classification after the custom adversarial training process. Please explore if it is possible, otherwise I will suggest another solution.