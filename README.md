# Signature Verification

## Description

The **Signature Verification** project is designed to compare two signatures and determine whether they match. This is achieved using a deep learning model based on Siamese networks, inspired by the approach described in the research paper [SigNet](https://arxiv.org/abs/1707.02131). The system classifies signatures as genuine or forged, providing a robust solution for signature authentication.

---

## Features

- **Deep Learning Model**: Uses a Siamese network for signature classification.
- **API for Signature Verification**: Accepts two signature files and provides a JSON response containing:
  - Prediction (Genuine or Forgery)
  - Similarity Distance
  - Confidence Percentage
  - Combined Image of the signatures
- Easy-to-use interface for training, testing, and API deployment.
- Modular design for flexibility and scalability.

---

## My Observations

- I tried to train the model on ICDAR train dataset first, using contrastive loss (euclidean distance) and the overall accuracy was around 67% tested on the ICDAR test data. However this didn't perform well on real world signatures. So, next I trained using CEDAR dataset using Contrastive Loss (Cosine similarity) and ended up overfitting multiple times despite adjusting learning rates utilizing multiple scheduler techniques such as ReduceLR. Next, I tried training the model using Contrastive Loss (Euclidean distance) and this yielded a good model at the end of Epoch 15. It worked quite well on real world signatures.
- Contrastive Loss - I believe that Euclidean Distance is more appropriate for the Siamese Network as opposed to Cosine Similarity, even though there is literature out there which says that Cosine similarity is better for comparing signatures.
- Learning rate - I find that keep the Learning rate at 1e-3 and using a StepLR scheduler to decay the learning rate by a factor of 0.5 every 3 epochs works well.
- Training HW - I used my NVIDIA laptop graphic card for training and used the CUDA library. I have adapted the code so that it works on both CPU and GPU.
- Given my limited time and resources, I believe that there is quite a bit of scope for improvement (for e.g. adding dropout, using a larger and more varied dataset, other loss functions, etc)
- I have used FastAPI and Streamlit wrapper. For temporarily hosting it to public networks, I used ngrok which seems like an extremely useful tool.



## File Structure

The project includes the following key files:

1. **`train.py`**  
   - Used to train the Siamese network on the dataset.  
   - Saves the trained model for later use.

2. **`test.py`**  
   - Evaluates the model on a dataset.  
   - Outputs metrics such as accuracy and other evaluation scores.

3. **`SigVerify.py`**  
   - Provides an API to accept two input signature images.  
   - Returns a JSON response with the prediction, similarity distance, confidence percentage, and combined image.

4. **`test_api.py`**  
   - Contains code to test the API functionality.

5. **`config.py`**  
   - Centralized configuration file for file paths and other project-related variables.

5. **`cedar_train and cedar_test CSV files`**  
   - I have created these CSV files to work with the CEDAR signature data

Additionally, the project requires a dataset for training and evaluation.  The dataset can be found on Kaggle [Kaggle Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)

---

## Usage

### Training the Model
Run the following command to train the Siamese network:  
```bash
python train.py
```

### Testing the Model
Evaluate the trained model using:  
```bash
python test.py
```

### Running the API
Start the API for signature verification:  
```bash
python SigVerify.py
```

### Testing the API
Use `test_api.py` to test the API functionality:  
```bash
python test_api.py
```

---

## Configuration

Modify the `config.py` file to update file paths and other variables as needed for your system setup.

---

## References

- Research Paper: [SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification](https://arxiv.org/abs/1707.02131)



## Disclaimer

This project is for my personal learning and not production grade. 

---
