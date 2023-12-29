<h2>
<p align='center'>
Guided Conditional Image Generation with Conditional Flow Matching
</p>
</h2>

<h4 align='center'> Project Description </h4>
The project innovatively integrates Conditional Optimal Transport into an attention-based UNet model for both conditional and unconditional image generation tasks. Utilizing a Classifier Free Guidance (CFG) mechanism ensures a unified model's proficiency across tasks. Addressing the descriptive limitations of the CIFAR10 dataset, the BLIP2 FLAN T5 model is employed for image captioning, enhancing the conditioning process. The self and cross attention mechanism, incorporating timestep and tokenized text, facilitates conditioning. Extensive experimental analysis leads to an optimized architecture with a FID score of 105.54 for unconditional generation and CLIPScore/FID scores of 22.19/305.42 for conditional generation. The research highlights the model's potential, suggesting further improvements through architectural refinements and extended training.

### Technical Skills 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
<br>

### Dependencies
##### Transformers
      !pip install transformers
##### PyTorch (Check CPU/GPU Compatibility)
      https://pytorch.org/get-started/locally/
##### Pandas
      !pip install pandas
##### NumPy
      !pip install numpy
##### Matplotlib
      !pip install matplotlib
##### TorchDiffEq
      !pip install torchdiffeq
##### Torchmetrics
      !pip install torchmetrics
##### Torchviz
      !pip install torchviz
##### Torch Fidelity
      !pip install torch-fidelity

### Dataset Information
* CIFAR-10
   * Publicly available at: https://www.cs.toronto.edu/~kriz/cifar.html
   * For the Caption Generation check `Caption_Generation.ipynb`
  
### File Content
* Caption_Generation.ipynb:
   - Utilizes the BLIP2 model to generate descriptive captions for images in the CIFAR dataset and stores the resulting dataset as a pickle file.

* Cross_Validation.ipynb:
   - Implements code for cross-validation using a list of learning rates.

* Flow_Matching_Training.ipynb:
   - Encompasses the entire training process, employing flow matching with a conditional optimal transport objective in conjunction with the proposed UNet model.

* Flow_Inference.ipynb:
   - Contains code for generating images from uniformly sampled inputs and evaluates the FID and CLIPScore metrics for the trained models.

* Text_Encoding.ipynb:
   - Utilizes the BLIP2 tokenizer to convert captions into tokens for subsequent use in the conditioning process.

* UNet_Attn.ipynb:
   - Houses the proposed UNet model, a key component in the conditional and unconditional image generation tasks.

* Docs
   * Project Report: Contains the documented project with the Problem Statement, Data Augmentation, Methodology, UNet Model, and the Results

### How to run
1. **Dependency Installation:**
   - Execute the command to install project dependencies necessary for proper functioning.

2. **Repository Cloning:**
   - Clone the project repository to the local machine using the command:
     ```
     git clone https://github.com/Anshumaan-Chauhan02/Guided-Flow-Matching
     ```

3. **Caption Generation:**
   - Run the `Caption_Generation.ipynb` notebook to generate a captioned dataset utilizing the BLIP2 model.

4. **Flow Matching Training:**
   - Execute the `Flow_Matching_Training.ipynb` notebook to initiate the training process for the unconditional/conditional generation model.

5. **Model Evaluation and Inference:**
   - Run the `Flow_Inference.ipynb` notebook for comprehensive model evaluation and generation of inferences.

**Note:**
   - Ensure to update the specified file paths in the notebooks with the appropriate local repository path.
