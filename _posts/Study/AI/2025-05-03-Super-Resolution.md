---
layout: post
title: Super-Resolution Study
date: 2025-05-03 09:27 +0100
categories: [Study, AI]
tags: [computer vision, machine learning, deep learning, super-resolution]
mermaid: true
math: true
pin: false
---

## Hitchhiker’s Guide to Super-Resolution: Introduction and Recent Advances

[▶ Original Report Link](https://ar5iv.labs.arxiv.org/html/2209.13131?_immersive_translate_auto_translate=1)

### 2 Setting and Terminology

#### 2.1 Problem Definition: Super-Resolution

Super-Resolution (SR) refers to methods that can **develop High-Resolution (HR) images** from **at least one Low-Resolution (LR) image**

##### 2.1.1 Single Image Super-Resolution (SISR)

Low-Resolution (LR) image 

$$x \in ℝ^{\bar{w} \times \bar{h} \times c}$$

High-Resolution (HR) image 

$$x \in ℝ^{w \times h \times c}$$

with $$\bar{w} \leq w$$ and $$\bar{h} \leq h$$

The amount of pixels of an image:

$$N_x = w \cdot h \cdot c$$

The set of all valid positions in 𝐱: 

$$\omega_x = \{(i,j,k) \in ℕ^3_1 \| i \leq h, j \leq w, k \leq c\}$$

A scaling factor:

$$s \in ℕ$$

It holds that $$h = s \cdot \bar{h}$$ and $$w = s \cdot \bar{w}$$

the inherent relationship between the two entities LR (𝐱) and HR (𝐲): 

$$𝒟: ℝ^{w \times h \times c} → ℝ^{\bar{w} \times \bar{h} \times c}$$  

$$x = 𝒟(𝐲;δ)$$ 

in which δ are parameters of 𝒟 that contain, for example, the scaling factor s and other elements like blur type.

#### 2.2 Evaluation: Image Quality Assessment (IQA)

##### 2.2.1 Mean Opinion Score (MOS)

**Human viewers** rate images with quality scores, typically 1 (bad) to 5 (good).

MOS is the arithmetic mean of all ratings. Despite **reliability**, mobilizing human resources is **time-consuming** and **cumbersome**, especially for large datasets.

##### 2.2.2 Peak Signal-to-Noise Ratio (PSNR)

It is the ratio between the maximum possible pixel-value L (255 for 8-bit representations) and the **Mean Squared Error (MSE)** of reference images. Given the approximation $\hat{𝐲}$ and the ground-truth 𝐲, PSNR is a logarithmic quantity using the decibel scale [dB]:

$$\mathrm{PSNR}\left(\mathbf{y},\widehat{\mathbf{y}}\right)=10\cdot\log_{10}\frac{L^2}{\frac{1}{N_{\mathbf{y}}}\sum_{p\in\Omega_{\mathbf{y}}}\left[\mathbf{y}_{p}-\widehat{\mathbf{y}}_{p}\right]^2}$$

It focuses on **pixel-level differences** instead of **mammalian visual perception**, which is more attracted to structures

It correlates **poorly with subjectively perceived quality**

##### 2.2.3 Structural Similarity Index (SSIM)

The Structural Similarity Index (SSIM) depends on three relatively independent entities: **luminance**, **contrast**, and **structures**

SSIM estimates for an image 𝐲 the **luminance** $$μ_𝐲$$ as the mean of the intensity, while it is estimating **contrast** $$σ_𝐲$$ as its standard deviation:

$$\mu_{\mathbf{y}}=\frac{1}{N_{\mathbf{y}}}\sum_{p\in\Omega_{\mathbf{y}}}\mathbf{y}_{p}$$  

$$\sigma_{\mathbf{y}}=\frac{1}{N_{\mathbf{y}}-1}\sum_{p\in\Omega_{\mathbf{y}}}\left[\mathbf{y}_{p}-\mu_{\mathbf{y}}\right]^{2}$$

A similarity comparison function S:

$$S
\begin{pmatrix}
x,y,c
\end{pmatrix}=\frac{2\cdot x\cdot y+c}{x^2+y^2+c},$$

where x and y are the compared scalar variables, and $$c = (k \cdot L)^2, 0 < k \ll 1$$ is a constant to avoid instability.

Given a ground-truth image 𝐲 and its approximation $$\hat{y}$$, the comparisons on luminance ($$𝒞_l$$) and contrast ($$𝒞_c$$) are

$$\mathcal{C}_l\left(\mathbf{y},\mathbf{\hat{y}}\right)=S\left(\mu_\mathbf{y},\mu_\mathbf{\hat{y}},c_1\right)\mathrm{~and~}\mathcal{C}_c\left(\mathbf{y},\mathbf{\hat{y}}\right)=S\left(\sigma_\mathbf{y},\sigma_\mathbf{\hat{y}},c_2\right)$$

where $$c_1, c_2 > 0$$. The empirical co-variance

$$\sigma_{\mathbf{y},\mathbf{\hat{y}}}=\frac{1}{N_{\mathbf{y}}-1}\sum_{p\in\Omega_{\mathbf{y}}}\left(\mathbf{y}_{p}-\mu_{\mathbf{y}}\right)\cdot\left(\mathbf{\hat{y}}_{p}-\mu_{\mathbf{\hat{y}}}\right),$$

determines the structure comparison ($$𝒞_s$$), expressed as the correlation coefficient between 𝐲 and $$\hat{y}$$:

$$\mathcal{C}_s\left(\mathbf{y},\widehat{\mathbf{y}}\right)=\frac{\sigma_{\mathbf{y},\hat{\mathbf{y}}}+c_3}{\sigma_{\mathbf{y}}\cdot\sigma_{\hat{\mathbf{y}}}+c_3},$$

where $$c_3 > 0$$. Finally, the SSIM is defined as:

$$\mathrm{SSIM}\left(\mathbf{y},\mathbf{\hat{y}}\right)=\left[\mathcal{C}_{l}\left(\mathbf{y},\mathbf{\hat{y}}\right)\right]^{a}\cdot\left[\mathcal{C}_{c}\left(\mathbf{y},\mathbf{\hat{y}}\right)\right]^{\beta}\cdot\left[\mathcal{C}_{s}\left(\mathbf{y},\mathbf{\hat{y}}\right)\right]^{\gamma}$$

where α>0,β>0 and γ>0 are adjustable control parameters for weighting relative importance of all components.

##### 2.2.4 Learning-based Perceptual Quality (LPQ)

In essence, LPQ tries to approximate a variety of **subjective ratings** by **applying DL methods**.

A significant drawback of LPQ is the limited **availability of annotated samples**.

##### 2.2.5 Task-based Evaluation (TBE)

One can focus on task-oriented features.

##### 2.2.6 Evaluation with defined Features

One example is the Gradient Magnitude Similarity Deviation (GMSD), which uses the pixel-wise Gradient Magnitude Similarity (GMS)

An alternative is the Feature Similarity (FSIM) Index. It also uses gradient magnitudes, but combines them with Phase Congruency (PC), a local structure measurement, as feature points.

##### 2.2.7 Multi-Scale Evaluation

In practice, SR models usually super-resolve to different scaling factors, known as Multi-Scaling (MS). Thus, evaluating metrics should address this scenario.

#### 2.3 Datasets and Challenges

Two of the most famous challenges are the New Trends in **Image Restoration** and **Enhancement (NTIRE) challenge**, and the Perceptual Image Restoration and Manipulation (PIRM) challenge.

#### 2.4 Color Spaces

Exploring other color spaces for DL-based SR methods is nearly nonexistent, which presents an exciting research gap.

### 3 Learning Objectives

#### 3.1 Regression-based Objectives

##### 3.1.1 Pixel Loss

The first one is **the Mean Absolute Error (MAE)**, or L​1-loss:

$$\mathcal{L}_{\mathrm{L1}}\left(\mathbf{y},\widehat{\mathbf{y}}\right)=\frac{1}{N_{\mathbf{y}}}\sum_{p\in\Omega_{\mathbf{y}}}\left|\mathbf{y}_{p}-\widehat{\mathbf{y}}_{p}\right|$$

It takes the absolute differences between every pixel of both images and returns the mean value.

The second well-known pixel loss function is the **Mean Squared Error (MSE)**, or L2-loss. It weights high-value differences higher than low-value differences due to an additional square operation:

$$\mathcal{L}_{\mathrm{L2}}\left(\mathbf{y},\mathbf{\hat{y}}\right)=\frac{1}{N_{\mathbf{y}}}\sum_{p\in\Omega_{\mathbf{y}}}\left|\mathbf{y}_{p}-\mathbf{\hat{y}}_{p}\right|^{2}$$

##### 3.1.2 Uncertainty-Driven Loss

An **adaptive weighted loss** for SISR, which aims at prioritizing texture and edge pixels that are visually more significant than pixels in smooth regions. Thus, the adaptive weighted loss treats every pixel unequally.

##### 3.1.3 Content Loss

Instead of using the difference between the approximated and the ground-truth image, one can transform both entities further into a more discriminant domain.

In more detail, the feature extractor is pre-trained on another task, i.e., image classification or segmentation. During the training of the actual SR model on the difference of feature maps, the parameters of the feature extractor remain fixed. Thus, the goal of the SR model is not to generate pixel-perfect estimations. Instead, it produces images whose features are close to the features of the target.

#### 3.2 Generative Adversarial Networks

The core idea is to use two distinct networks: a generator G and a discriminator D. The generator network learns to produce samples close to a given dataset and to fool the discriminator.

##### 3.2.1 Total Variation Loss

One way to regularize GANs is to use a Total Variation (TV) denoising technique known from image processing.

$$\mathrm{TV}(\mathbf{y})=\frac{1}{N_{\mathbf{y}}}\sum_{i,j,k}\sqrt{\underbrace{\left(\mathbf{y}_{i+1,j,k}-\mathbf{y}_{i,j,k}\right)^2+\underbrace{\left(\mathbf{y}_{i,j+1,k}-\mathbf{y}_{i,j,k}\right)^2}_{\text{diff. first axis}}}_{\text{diff. second axis}}}$$

##### 3.2.2 Texture Loss

Texture synthesis with parametric texture models has a long history with the goal of transferring global texture onto other images

#### 3.3 Denoising Diffusion Probabilistic Models

**Denoising Diffusion Probabilistic Models (DDPMs)** exploit this insight by formulating a Markov chain to alter one image into a noise distribution gradually, and the other way around.

### 4 Upsampling

#### 4.1 Interpolation-based Upsampling

Many DL-based SR models use image interpolation methods because of their simplicity. The most known methods are **nearest-neighbor**, **bilinear**, and **bicubic interpolation**.

#### 4.2 Learning-based Upsampling

##### 4.2.1 Transposed Convolution

Transposed convolution expands the spatial size of a given feature map and subsequently applies a convolution operation.

##### 4.2.2 Sub-Pixel Layer

Introduced with ESPCN, it uses a convolution layer to extract a deep feature map and rearranges it to return an upsampled output.

##### 4.2.3 Decomposed Upsampling

An extension to the above approaches is decomposed transposed convolution. Using 1D convolutions instead of 2D convolutions reduces the number of operations and parameters for the component $$k^2$$ to 2⋅k.

##### 4.2.4 Attention-based Upsampling

Another alternative to transposed convolution is attention-based upsampling [69]. It follows the definition of attention-based convolution (or scaled dot product attention) and replaces the 1x1 convolutions with upsampling methods.

##### 4.2.5 Upsampling with Look-Up Tables

Before generating the LUT, a small-scale SR model is trained to upscale small patches of a LR image to target HR patches. Subsequently, the LUT is created by saving the results of the trained SR model applied on a uniformly distributed input space. It reduces the upsampling runtime to the time necessary for memory access while achieving better quality than bicubic interpolation. On the other hand, it requires additional training to create the LUT.

##### 4.2.6 Flexible Upsampling

In order to overcome this limitation, a meta-upscale module was proposed [41]. It predicts a set of filters for each position in a feature map that is later applied to a location in a lower-resolution feature map. 

### 5 Attention Mechanisms for SR

#### 5.1 Channel-Attention

Feature maps generated by CNNs are not equally important. Therefore, essential channels should be weighted higher than counterpart channels, which is the goal of channel attention. It focuses on “which” (channels) carry crucial details.

#### 5.2 Spatial-Attention

In contrast to channel attention, spatial attention focuses on “where” the input feature maps carry important details, which requires extracting global information from the input.

#### 5.3 Mixed Attention

Since both attention types can be applied easily, merging them into one framework is natural. Thus, the model focuses on “which” (channel) is essential and “where” (spatially) to extract the most valuable features. This combines the benefits of both approaches and introduces an exciting field of research, especially in SR. One potential future direction would be to introduce attention mechanisms incorporating both concerns in one module.

### 6 Additional Learning Strategies

#### 6.1 Curriculum Learning

Curriculum learning follows the idea of training a model under easy conditions and gradually involving more complexity [84], i.e., additional scaling sizes.

#### 6.2 Enhanced Predictions

Instead of enhancing simple input-output pairs, one can use data augmentation techniques like rotation and flipping for final prediction.

#### 6.3 Learned Degradation

The Content Adaptive Resampler (CAR) introduced a resampler for downscaling. It predicts kernels to produce downscaled images according to its HR input. Next, a SR model takes the LR image and predicts the SR image. Thus, it simultaneously learns the degradation mapping and upsampling task.

#### 6.4 Network Fusion

Network fusion uses the output of all additional SR models and applies a fusion layer to the outputs. Finally, it predicts the SR image used for the learning objective.

#### 6.5 Multi-Task Learning

E.g., one can assign a label to each image and use multiple datasets for training. Next, a SR model can learn to reconstruct the SR image and predict its category (e.g., natural or manga image)

#### 6.6 Normalization Techniques

A slight change in the input distribution is a cause of many issues because layers need to continuously adapt to new distributions, which is known as covariate shift and can be alleviated with BatchNorm.

### 7 SR Models

### 8 Unsupervised Super-Resolution

#### 8.1 Weakly-Supervised

Weakly-supervised methods use unpaired LR and HR images

The first generator takes a LR image and super-resolves it. The output of the first generator constitutes a SR image

The second generator takes the prediction of the first generator and performs the inverse mapping. The result of the second generator is optimized via content loss with the original input, the LR image.

#### 8.2 Zero-Shot

Zero-shot or one-shot learning is associated with training on objects and testing on entirely different objects from a different class that was never observed.

#### 8.3 Deep Image Prior

It uses a CNN to predict the LR image when downsampled, given some random noise instead of an actual image. Therefore, it follows the strategy of ZSSR by using only the LR image. However, it fixes the input to random noise and applies a fixed downsampling method to the prediction.

### 9 Neural Architecture Search

[▶ Original Report Link](https://ar5iv.labs.arxiv.org/html/2205.07514?_immersive_translate_auto_translate=1)

## Residual Local Feature Network for Efficient Super-Resolution

### 2 Related Work

#### 2.1 Efficient Image Super-Resolution

SCRNN applied the deep learning algorithm to the SISR field for the first time. It has three layers and uses bicubic interpolation to upscale the image before the net, causing unnecessary computational cost.

To address this issue, FSRCNN employed the deconvolution layer as the upsampling layer and upscaled the image at the end of net. 

DRCN introduced a deep recursive convolutional network to reduce the number of parameters. 

LapSRN proposed the laplacian pyramid super-resolution block to reconstruct the sub-band residuals of HR images.

CARN proposed an efficient cascading residual network with group convolution, which obtains comparable results against computationally expensive models.

IMDN proposed a lightweight information multi-distillation network by constructing the cascaded information multi-distillation blocks, which extracts hierarchical features step-by-step with the information distillation mechanism (IDM).

RFDN refined the architecture of IMDN and proposed the residual feature distillation network, which replaced IDM with feature distillation connections.

ECBSR proposed an edge-oriented convolutional block based on the reparameterization technique[10], which can improve the learning ability of the model without increasing the inference time.

#### 2.2 Train Strategy for PSNR-oriented SISR

These SR networks are usually trained by the ADAM optimizer with standard l1 loss for hundreds of epoches. To improve the robustness of training, they usually adopt a smaller learning rate and patch size.

Recent works on image recognition[3] and optical flow estimation[41] have demonstrated that advanced training strategies can enable older network architectures to match or surpass the performance of novel architectures.

RFDN[31] demonstrated that both fine-tuning the network with l2 loss and initializing a 4x SR model with pretrained 2x model can effectively improve PSNR. 

RRCAN[30] revisited the popular RCAN model and demonstrated that increasing training iterations clearly improves the model performance.

### 3 Method

#### 3.1 Network Architecture

```mermaid
graph LR
  A[Input: Low-Resolution Image] --> B[Conv 3×3<br>Shallow Feature Extraction]
  B --> C1[RLFB Block 1]
  C1 --> C2[RLFB Block 2]
  C2 --> C3[RLFB Block 3]
  C3 --> D[Conv 3×3<br>Post RLFB]

  B -.-> E[Skip Connection]
  D --> F[Add: Long Residual]
  E --> F

  F --> G[Conv 3×3<br>Upsample Preparation]
  G --> H[PixelShuffle<br>Upsampling]
  H --> I[Output: High-Resolution Image]
```

#### 3.2 Revisiting the Contrastive Loss

 The contrastive loss is defined as:

#### 3.3 Warm-Start Strategy

For large scale factors like 3 or 4 in the SR task, some previous works use **the 2x model** as **a pre-trained network** instead of training them from scratch.

## Efficient Long-Range Attention Network for Image Super-resolution

[▶ Original Report Link](https://ar5iv.labs.arxiv.org/html/2203.06697?_immersive_translate_auto_translate=1)

### 2 Related work

#### 2.1 CNN-based SR methods

CNN-based methods have demonstrated impressive performance in the SR task. 

To build more effective models for SR, the recently developed methods tend to employ deeper and more complicated architectures as well as the attention techniques.

#### 2.2 Transformer-based SR methods

The breakthrough of transformer networks in natural language processing (NLP) inspired of use of self-attention (SA) in computer vision tasks. The SA mechanism in transformers can effectively model the dependency across data, and it has achieved impressive results on several high-level vision tasks, such as image classification, image detection, and segmentation. 

### 3 Methodology

#### 3.1 Overall Pipeline of ELAN

```mermaid
graph LR
    A[Input: Low-Resolution Image] --> B[Conv 3×3<br>Shallow Feature Extraction]
    B --> C1[ELAB 1]
    C1 --> C2[ELAB 2]
    C2 --> C3[ELAB 3]
    C3 --> C4[ELAB 4]
    C4 --> C5[ELAB 5]
    C5 --> D[Conv 3×3<br>Reconstruction]
    D --> E[PixelShuffle]
    E --> F[Output: High-Resolution Image]
```

```mermaid
graph LR
    X[Input Feature] --> A1[ShiftConv]
    A1 --> A2[ReLU]
    A2 --> A3[ShiftConv]
    A3 --> A4[GMSA<br>Group-wise Self-Attention]
    A4 --> Y[Output Feature]
```

```mermaid
graph LR
    G1[Input] --> Split[Split into Groups]
    Split --> Shift[Shift-Convolution per Group]
    Shift --> ASA1[ASA Block]
    Shift --> ASA2[...]
    ASA1 --> Merge[Concatenate & Merge]
    Merge --> Out[Output]
```

#### 3.2 Efficient Long-range Attention Block (ELAB)

```mermaid
graph TD
    A[Input] --> Theta_i[Theta_i]
    A --> G_i[G_i]

    Theta_i --> R1[Reshape]
    G_i --> R2[Reshape]

    R1 --> M1[Multiplication]
    R2 --> M1

    M1 --> R3[Reshape]

    Theta_next[Theta_i_plus_n] --> R4[Reshape]
    G_next[G_i_plus_n] --> R5[Reshape]

    R4 --> M2[Multiplication]
    R5 --> M2

    M2 --> R6[Reshape]

    R3 --> FinalOut1[Output Part 1]
    R6 --> FinalOut2[Output Part 2]
```
```mermaid
graph LR
    A[Input] --> Theta_i[Shared Theta_i]
    A --> G_i[G_i]

    Theta_i --> R1[Reshape]
    G_i --> R2[Reshape]

    R1 --> M1[Multiplication]
    R2 --> M1

    M1 --> R3[Reshape]

    G_next[G_i_plus_n] --> R4[Reshape]
    R1 --> M2[Multiplication Shared]
    R4 --> M2

    M2 --> R5[Reshape]

    R3 --> FinalOut1[Output Part 1]
    R5 --> FinalOut2[Output Part 2]
```

## ShuffleMixer: An Efficient ConvNet for Image Super-Resolution  

[▶ Original Report Link](https://ar5iv.labs.arxiv.org/html/2205.15175?_immersive_translate_auto_translate=1)

### Proposed Method

We aim to develop an efficient large-kernel CNN model for the SISR task. To meet the efficiency goal, we introduce key designs to the feature mixing block employed to encode information efficiently.

#### 3.1 ShuffleMixer Architecture

```mermaid
graph LR
    A[Input Image] --> B[Feature Extraction]
    B --> C[FMB 1 - Feature Mixing Block]
    C --> D[FMB 2 - Feature Mixing Block]
    D --> E[... more FMBs]
    E --> F[Upsampler - Bilinear Upsampling]
    F --> G[Conv 3x3 - Reconstruction]
    G --> H[Output Image]

    %% Optional skip connection
    A -. Skip Connection .-> H
```

```mermaid
graph TD
    Z0[Input Z0] --> L1[Shuffle Mixer Layer 1]
    L1 --> L2[Shuffle Mixer Layer 2]
    L2 --> FMBConv[Conv Layer - FMBConv]
    FMBConv --> Add[Residual Add with Z0]
    Add --> Z_out[Output]

    SM_Input[SM Layer Input] --> P1[Channel Projection 1x1]
    P1 --> DWC[Depthwise Convolution]
    DWC --> P2[Channel Projection 1x1]
    P2 --> SM_Output[SM Layer Output]

    Z0_main[Input Feature Z0] --> LN[LayerNorm]
    LN --> Split[Channel Splitting]
    Split --> MLP[Point-wise MLP Branch]
    Split --> ID[Identity Branch]
    MLP --> Merge1
    ID --> Merge1
    Merge1[Channel Shuffle] --> SM2[Next Shuffle Mixer Layer]

    X[Input Tensor] --> C1[Conv 1x1]
    C1 --> Act[SiLU Activation]
    Act --> C2[Conv 1x1]
    C2 --> Out_MLP[Output Tensor]

    X2[Input Tensor] --> F1[Conv 3x3]
    F1 --> Act2[SiLU Activation]
    Act2 --> F2[Conv 1x1]
    F2 --> Out_MB[Output Tensor]
```

## Omni Aggregation Networks for Lightweight SR

[▶ Original Report Link](https://ar5iv.labs.arxiv.org/html/2304.10244?_immersive_translate_auto_translate=1)

### 3 Methodology

#### 3.1 Attention Mechanisms in Super-Resolution

**Two** attention paradigms are widely adopted in SR to assist in analyzing and aggregating comprehensive patterns.

**Spatial Attention**. Spatial attention can be regarded as an anisotropic selection process. Spatial self-attention and spatial gate are predominantly applied.

**Channel Attention**. There are two categories of channel attention, i.e., scalar-based and covariance-based, proposed to perform channel recalibration or transmit patterns among channels.

#### 3.2 Omni Self-Attention Block

To mine all the correlations hidden in the latent variables, we propose a novel self-attention paradigm called Omni Self-Attention (OSA) block. Unlike existing self-attention paradigms (e.g., spatial self-attention [51, 5, 37]) that only indulge in unidimensional processing, OSA establishes the spatial and channel context simultaneously. 

The proposed OSA calculates the score matrices corresponding to the space and channel direction through sequential matrix operations and rotation

$$\begin{aligned}
 & Q^{s}=X\cdot W_{q},\quad K^{s}=X\cdot W_{k},\quad V^{s}=X\cdot W_{v}, \\
 & Y_{s}=\mathcal{A}^{s}(Q^{s},K^{s},V^{s})=\mathrm{SoftMax}(Q^{s}K^{sT})\cdot V^{s}, \\
 & Q^{c}=\mathcal{R}(Q^{^{\prime}}),K^{c}=\mathcal{R}(K^{^{\prime}}),V^{c}=\mathcal{R}(V^{^{\prime}}), \\
 & Y_{c}=\mathcal{A}^{c}(Q^{c},K^{c},V^{c})=\mathrm{SoftMax}(K^{c}Q^{cT})\cdot V^{c}, \\
 & Y_{OSA}=\mathcal{R}^{-1}\left(Y_{c}\right),
\end{aligned}$$

#### 3.3 Omni-Scale Aggregation Group

OSAG mainly consists of three stages: local, meso and global aggregations.

![](/assets/img/AI/Super%20Resolution/Omni-Scale%20Aggregation%20Group.png)

#### 3.4 Network Architecture

Omni-SR consists of three parts, i.e., **shallow feature extraction**, **deep feature extraction**, and **image reconstruction**.

**Omni-Scale Aggregation Group (OSAG).**

Each OSAG contains **a local convolution block (LCB)**, **a meso-OSA block**, **a global-OSA block**, and **an ESA block**

**Optimization Objective**

Following prior works, we train the model by minimizing a standard $$L_1$$ loss between model prediction $$\hat{I}_{H​R}$$ and HR label $$I_{H​R}$$ as follows:

$$\mathcal{L}=\|I_{HR}-\widehat{I}_{HR}\|_{1}.$$

## Equivalent Transformation and Dual Stream Network Construction for Mobile Image Super-Resolution

[▶ Original Report Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Chao_Equivalent_Transformation_and_Dual_Stream_Network_Construction_for_Mobile_Image_CVPR_2023_paper.pdf)


## Table of Reports

Related Search tag in `Google Scholar`:

```
"image super-resolution" AND ("lightweight" OR "efficient") AND ("dynamic" OR "adaptive") after:2021
```

| **Title** (with Year)                                                                                    | **Tags**                                                   | **Summary**                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Lightweight Image Super-Resolution with Adaptive Weighted Learning Network(IMDN) (2019)**              | *Lightweight, Information Distillation, Channel Attention* | IMDN introduces an Information Multi-distillation Network that employs a series of distillation blocks to extract and fuse hierarchical features efficiently. The network utilizes contrast-aware channel attention mechanisms to enhance feature representation, achieving a balance between performance and computational efficiency.                                                                                        |
| **Exploring Sparsity in Image Super-Resolution for Efficient Inference (SMSR) (2021)**                   | *Sparse Mask, Dynamic Inference, Efficient*                | SMSR introduces a Sparse Mask Super-Resolution network that learns spatial and channel-wise masks to identify and skip redundant computations during inference. While it enhances efficiency by pruning unnecessary operations, the binary nature of the masks limits fine-grained enhancements in complex regions.                                                                                                            |
| **Dynamic Residual Self-Attention Network for Lightweight Single Image Super-Resolution (DRSAN) (2021)** | *Dynamic, Self-Attention, Lightweight*                     | DRSAN proposes a dynamic residual self-attention mechanism that adaptively adjusts residual connections based on input content. This approach allows the network to modulate feature strength dynamically; however, the overall network structure remains static, lacking dynamic routing capabilities for different content complexities.                                                                                     |
| **Hitchhiker’s Guide to Super-Resolution: Introduction and Recent Advances (2023)**                      | *Survey, Overview*                                         | Comprehensive survey of deep learning-based SR, covering recent advancements like transformer-based models and diffusion approaches, and discussing remaining challenges (e.g., flexible upsampling, new loss functions, better metrics)..                                                                                                                                                                                     |
| **Residual Local Feature Network for Efficient SR (RLFN) (2022)**                                        | *Lightweight, Efficient*                                   | Proposes a compact network using only three conv layers for local feature learning, simplifying feature aggregation for a better speed–quality trade-off. With an improved contrastive loss and multi-stage warm-start training, RLFN outperforms prior efficient models in runtime while maintaining high PSNR/SSIM.                                                                                                          |
| **Efficient Long-Range Attention Network (ELAN) (2022)**                                                 | *Efficient, Attention, Transformer*                        | Introduces an efficient transformer-inspired SR model that reduces self-attention cost. Uses shift convolution to extract local structure and a group-wise multi-scale self-attention (GMSA) module to capture long-range dependencies. An efficient long-range attention block (ELAB), built by combining shift-conv and GMSA with shared attention, achieves transformer-level accuracy with significantly fewer parameters. |
| **ShuffleMixer: An Efficient ConvNet for SR (2022)**                                                     | *Lightweight, Efficient, ConvNet*                          | Proposes a lightweight SR network with large depth-wise convolutions and channel split-shuffle operations to efficiently enlarge the receptive field. Augments large kernels with fused MobileNet-like convolution blocks to preserve fine details. Experimental results show ShuffleMixer is \~3× smaller (in FLOPs and params) than previous efficient models (e.g., CARN) while achieving competitive performance.          |
| **Omni-SR: Omni Aggregation Networks for Lightweight SR (2023)**                                         | *Lightweight, Transformer, Multi-scale*                    | Transformer-based framework targeting <1M parameters. Introduces an **Omni Self-Attention (OSA)** block that densely connects spatial and channel attention, simultaneously modeling correlations across spatial and channel dimensions. Also proposes a multi-scale interaction scheme to expand the effective receptive field in shallow models. Omni-SR achieved record-high performance for lightweight SR                 |
| **ETDS: Equivalent Transformation & Dual-Stream Network for Mobile SR (2023)**                           | *Mobile, Dual-Branch, Efficient*                           | Targets real-time SR on mobile devices. Proposes an **Equivalent Transformation (ET)** technique to convert heavy layers into mobile-friendly ops (reparameterizing into standard conv+ReLU). Builds a dual-stream network to mitigate the extra parameters from ET and enhance feature extraction.                                                                                                                            |
| **SAFMN: Spatially-Adaptive Feature Modulation for Efficient SR (2023)**                                 |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **DLGSANet: Dynamic Local/Global Self-Attention Network (2023)**                                         |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **AsConvSR: Fast SR Network with Assembled Convolutions (2023)**                                         |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **WBSR: Weight-Balancing for Efficient SR (2024)**                                                       |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **PCSR: Pixel-Level Classification for SR (2024)**                                                       |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **LAMNet: Linear Adaptive Mixer Network for SR (2023)**                                                  |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **FIWHN: Feature Interaction Weighted Hybrid Network (2024)**                                            |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **TCSR: Transformer-CNN for Lightweight SR (2023)**                                                      |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **HFFN: High-Frequency Focused Network (2023)**                                                          |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **RAMiT: Reciprocal Attention Mixing Transformer (2023)**                                                |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **SCNet: Fully 1×1 Convolutional SR Network (2023)**                                                     |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **QSRNet: Quick SR Network for Mobile (2023)**                                                           |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **DITN: Deployment-Friendly Transformer for SR (2023)**                                                  |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Dynamic Graph SR via Message Passing & Conv Mixer (2025)**                                             |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **BSRN: Blueprint Separable Residual Network (2022)**                                                    |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **ECBSR: Edge-Oriented Convolution Block for Real-Time SR (2022)**                                       |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **ARM: Any-Time Super-Resolution Method (2022)**                                                         |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **ADSRNet: Adaptive Heterogeneous CNN for SR (2024)**                                                    |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **LDEN: Learnable Detail Enhancement Network (2025)**                                                    |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Swin2SR: SwinV2 Transformer for Compressed SR (2023)**                                                 |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **ESRT: Efficient SR Transformer (2022)**                                                                |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |
|                                                                                                          |                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                |

## Possible Architecture

```shell
py -3.11 train2.py --train_dir Data/DIV2K_train_HR --val_dir Data/DIV2K_valid_HR --scale 2 --epochs 100 --batch_size 16 --patch_size 128
```

### Overall Architecture
```mermaid
flowchart TD
    A[Input LR Image] --> B[Conv 3×3, 3→C channels<br/>+ ReLU]
    B --> C[Shallow Features F₀]
    C --> D1[RDB Block 1]
    D1 --> D2[RDB Block 2]
    D2 --> D3[…]
    D3 --> Dn[RDB Block N]
    Dn --> E[Concat all block outputs]
    E --> F[Conv 1×1 Fusion → C channels]
    F --> G[ESA: Spatial Attention]
    G --> H[Add skip connection<br/>F + F₀]
    H --> I[Conv 3×3 → 3×scale² channels]
    I --> J[PixelShuffle ×scale]
    J --> K[Output HR Image]
```

### Residual Distillation Block

```mermaid
flowchart TD
    X[Input Features X] --> P1[Conv 3×3 → C channels<br/>+ ReLU] 
    P1 --> P2[Intermediate Feature 1]
    
    P2 --> P3[Conv 3×3 → C channels<br/>+ ReLU]
    P3 --> P4[Intermediate Feature 2]
    
    P4 --> P5[Conv 3×3 → C channels<br/>+ ReLU]
    P5 --> P6[Intermediate Feature 3]
    
    subgraph Fuse & Attention
        P2 --> Q1
        P4 --> Q1
        P6 --> Q1
        Q1[Concat F1,F2,F3] --> R[Conv 1×1 → C channels]
        R --> S[CALayer: Channel Attention]
    end
    
    S --> T[Add Residual S + X]
    T --> Y[Output Features]
```
