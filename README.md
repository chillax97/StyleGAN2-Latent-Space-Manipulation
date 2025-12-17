# Latent Space Manipulation with StyleGAN2-ADA

## Project Overview
This project investigates semantic control in generative models by manipulating latent spaces in **StyleGAN2-ADA**. The goal was to isolate and edit a single facial attribute—**smiling**—while preserving identity and other visual features. The study compares **Z-space** and **W-space** representations and evaluates different methods for discovering meaningful latent directions.

## Key Contributions
- Trained a **ResNet18** binary classifier on **CelebA** to detect the *Smiling* attribute with **93.32% validation accuracy**.
- Generated **5,000 synthetic faces** using StyleGAN2-ADA and labeled them automatically via the trained classifier.
- Extracted and stored both **Z-space** and **W-space** latent vectors for downstream analysis.
- Computed semantic direction vectors using:
  - **Mean Difference**
  - **Logistic Regression (hyperplane normal)**
- Demonstrated controlled attribute editing via  
  **v_new = v_old + α · n**, with systematic analysis of α scaling.
- Empirically showed that **W-space is significantly more disentangled** than Z-space.

## Technical Stack
- **Deep Learning:** PyTorch, StyleGAN2-ADA, ResNet18  
- **Data:** CelebA  
- **ML Methods:** Logistic Regression, Binary Classification  
- **Tools:** NumPy, pandas, scikit-learn, PIL  

## Methodology Summary
1. **Model Setup**
   - Cloned and configured StyleGAN2-ADA.
   - Used pretrained FFHQ weights for high-quality face synthesis.

2. **Attribute Classifier**
   - Fine-tuned ResNet18 (ImageNet pretrained).
   - Binary output with BCEWithLogitsLoss.

3. **Data Collection**
   - Sampled 5,000 random latent vectors.
   - Generated images and corresponding Z and W representations.
   - Labeled images using the trained classifier.

4. **Latent Direction Discovery**
   - Mean Difference: class-wise centroid subtraction.
   - Logistic Regression: separating hyperplane normal.
   - Normalized all direction vectors to ensure stable manipulation.

5. **Experiments**
   - Compared Z-space vs W-space edits.
   - Compared Mean Difference vs Logistic Regression.
   - Explored failure cases with extreme α values.

## Results & Insights
- **W-space edits** preserved identity and background far better than Z-space.
- **Z-space manipulations** exhibited strong entanglement (hair, background, extra faces).
- **Logistic Regression directions** produced cleaner and more realistic edits than mean difference.
- Extreme α values pushed samples off the learned manifold, causing visual artifacts.

## Key Takeaways
- Intermediate latent spaces (W-space) enable precise semantic control.
- Linear classifiers are effective tools for discovering meaningful directions in high-dimensional latent spaces.
- Proper normalization is critical for stable and interpretable latent manipulation.

## Author
**Osman Baha Sert**  
