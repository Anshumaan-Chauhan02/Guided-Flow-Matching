<h2>
<p align='center'>
Guided Conditional Image Generation with Conditional Flow Matching
</p>
</h2>

<h4 align='center'> Project Description </h4>
The project innovatively integrates Conditional Optimal Transport into an attention-based UNet model for both conditional and unconditional image generation tasks. Utilizing a Classifier Free Guidance (CFG) mechanism ensures a unified model's proficiency across tasks. Addressing the descriptive limitations of the CIFAR10 dataset, the BLIP2 FLAN T5 model is employed for image captioning, enhancing the conditioning process. The self and cross attention mechanism, incorporating timestep and tokenized text, facilitates conditioning. Extensive experimental analysis leads to an optimized architecture with a FID score of 105.54 for unconditional generation and CLIPScore/FID scores of 22.19/385.56 for conditional generation. The research highlights the model's potential, suggesting further improvements through architectural refinements and extended training.
