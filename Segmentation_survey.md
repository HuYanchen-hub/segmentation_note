# 1. Transformer-Based Visual Segmentation: A Survey 

## Abstract

## 1. Introduction
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307012336308.png)
## 2. Background
### 2.1 Problem Definition
semantic segmentation (SS)：类别可能是前景目标（thing）或者背景（stuff），每一类只有一个二值化mask表示像素属于该类。每一个mask不与其他mask重叠。
instance segmentation (IS)：每个类可能有一个以上的二值掩码，并且所有的类都是前景对象。一些IS掩码可能与其他掩码重叠。
panoptic segmentation (PS)：根据类的定义，每个类可能有不同数量的掩码。对于可数thing类，每个类对于不同的实例可能有多个掩码。对于不可数的stuff类，每个类只有一个面具。每个PS掩模与其他掩模不重叠。
![image.png|450](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307012347843.png)

### 2.2 Datasets and Metrics
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307012356865.png)
SS: mIOU
IS:  mAP
PS: panoptic quality(PQ)   **unifies both thing and stuff prediction by setting a fixed threshold 0.5**

> **分割质量**（segmentation quality，SQ）和**识别质量**（recognition quality)，RQ）的乘积
> $$PQ = \frac{\sum_{(p, q)\in TP}IoU(p, q)}{|TP| + \frac12|FP|+\frac12|FN|}$$
>  $$PQ = \frac{\sum_{(p, q)\in TP}IoU(p, q)}{|TP|} \times \frac{|TP|}{|TP| + \frac12|FP|+\frac12|FN|} $$
### 2.3 Segmentation Approaches Before Transformer
在ViT和DETR出现之前，正如FCN最初提出的那样，SS通常被作为一个稠密的像素分类问题来处理。
* **Semantic Segmentation.**
> better encoder-decoder frameworks
> larger kernels
> multiscale pooling
> multiscale feature fusion
> non-local modeling
> efficient modeling
> better boundary delineation
> design the variants of self-attention operators to replace the CNN prediction heads
* **Instance Segmentation**
> top-down approaches
> bottom-up approaches
> using gird representation to learn instance masks directly.
* **Panoptic Segmentation**  mainly focus on how to better fuse the results of both SS and IS
>top-down approaches
>bottom-up approaches
### 2.4 Transformer Basics

## 3. METHODS: A SURVEY
### 3.1 Meta-Architecture

* **Backbone**: 
>  ResNet50
>  explored the combination of CNNs with self-attention layers to model long-range dependencies(Non-local、Axial-DeepLab)
>  ViTs
* **Neck**
> FPN
> Deformable-DETR: deformable FPN
> Lite-DETR: 
* **Object Query**
* **Transformer Decoder**
> 分类：精化后的查询通过一个线性层直接映射到类预测。
> 检测：FFN预测目标包围盒的归一化中心坐标、高度和宽度。
> 分割：输出嵌入用于执行与特征F的点积，从而产生二进制掩码logits。
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307022142883.png)
* **Mask Prediction Representation**
> pixel-wise prediction as FCNs          用于语义感知的分割任务
> per-mask-wise prediction as DETR  用于实例感知的分割任务
* **Bipartite Matching and Loss Function**
> bipartite matching
> binary cross-entropy loss and dice loss

### 3.2 Method Categorization
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307022158125.png)
![image.png|800](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307022200409.png)

#### 3.2.1 Strong Representations
SETR  用ViT骨干网替换CNN骨干网 取得了当时的SOTA
* **Better ViTs Design**
> DeiT     提出知识蒸馏并提供强大的数据增强来高效地训练ViT
> MViT-V1  引入多尺度特征表示和池化策略来降低MHSA中的计算开销
> MViT-V2  进一步在MViT - V1中融合分解的相对位置嵌入和残差池化设计，从而获得更好的表示。
> MPViT     引入多尺度块嵌入和多路径结构，联合探索不同尺度的表征
> XCiT        跨特征通道而非令牌输入进行操作，并提出了交叉协方差注意力，其在令牌数量上具有线性复杂度。这种设计使得它很容易适应分割任务，因为它总是有高分辨率的输入。
> Pyramid ViT是第一个为检测和分割任务构建多尺度特征的工作。
* **Hybrid CNNs/Transformers/MLPs.**
> 许多工作不是修改ViT，而是将局部偏置引入ViT或直接使用大核CNN。
> Swin Transformer  v1、v2   采用CNN风格的偏移窗口注意力。还将模型扩展到大尺寸
> Segformer 设计了一种轻量级的transformer encoder。它包含MHSA过程中的序列缩减和一个轻量级的MLP解码器, 为SS实现了较好的速度和精度折中。
> 一些工作直接用一个transformer添加CNN层来探索局部上下文。
> 纯MLPs
> DWNet   提出动态深度卷积dynamic depth-wise convolution
> ConvNeXt  the larger kernel depthwise convolution      a stronger data training pipeline
> SegNext    设计CNN-like backbone with linear self-attention
> Meta-Former   ViT的元结构是取得更强效果的关键。这样的元结构包含一个令牌混合器、一个MLP和残差连接。令牌混合器默认为ViTs中的MHSA操作。Meta - Former表明令牌混合器并不像元结构那样重要。使用简单的池化作为令牌混合器可以获得更强的效果。
>explore the MLP-like architecture for dense prediction
* **Self-Supervised Learning (SSL)**
> MoCo-v3 冻结了patch projection layer以稳定训练过程。
> BERT-like pertaining (Mask Image Modeling, MIM) 
> MAE
> MaskFeat   研究MIM框架的重构目标如HOG特征.  
> 改进MIM框架或者将ViTs的主干替换为CNN架构
> VLM
#### 3.2.2 Interaction Design in Decoder
for improved cross-attention design in image segmentation   关注设计一个更好的解码器对原DETR的原解码器进行精化
for spatial-temporal cross-attention design in video segmentation将基于查询的目标检测器和分割器扩展到视频域，重点对时序一致性和关联性进行建模
* Improved Cross-Attention Design
> 目前针对改进交叉注意力的解决方案主要集中在`设计新的或增强的交叉注意力算子`和`改进解码器架构`上。
> `Deformable DETR`    提出可变形注意力来高效地采样点特征，并与对象查询联合执行交叉注意力。
> 有几项工作将对象查询引入到之前的RCNN框架中
> 
>    `Sparse-RCNN`  利用RoI池化特征对目标查询进行精化，进行目标检测。他们还提出了一种新的动态卷积和自注意力来增强对象查询，而不需要额外的交叉注意力。特别地，池化查询特征为对象查询重加权，然后将自注意力应用于对象查询以获得全局视图。
> 
> 一些工作增加了额外的掩模头用于实例分割
> QueryInst    添加掩码头并使用动态卷积细化掩码查询
> 一些工作通过在共享查询上直接应用MLP扩展了Deformable DETR
> 
> 受MEInst启发，SOLQ通过MLP在对象查询中使用掩码编码， 然而，这些工作仍然需要额外的盒子监督，这使得系统变得复杂。此外，由于掩模分辨率被限制在盒子内，大多数基于RoI的IS方法都存在掩模质量低的问

> 为了解决额外的盒子头问题，一些工作移除了盒子预测，并采用了纯粹的基于掩码的方法。直接从高分辨率特征中生成分割掩膜。
> Max-Deeplab  是第一个去除box head，设计基于纯掩膜的PS分割器。与基于box的PS方法相比，也取得了较强的性能。它结合了CNN - Transformer混合编码器和Transformer解码器作为额外的路径。Max - Deeplab仍然需要额外的辅助损失函数，如语义分割损失、实例判别损失等。
> ` K-Net` 使用掩码池化对掩码特征进行分组，并设计了门控动态卷积来更新相应的查询。通过`将分割任务看作不同核的卷积`，K - Net率先将所有三种图像分割任务统一起来，包括SS、IS和PS。
> `MaskFormer` 对原始DETR进行了扩展，去掉了box head，通过MLPs将对象查询转换为掩码查询。这说明简单的掩膜分类对于3种分割任务都能很好地工作。与MaskFormer相比，KNet具有良好的训练数据效率。这是因为K - Net采用掩码池化来定位对象特征，然后相应地更新对象查询。
>    受此启发，`Mask2Former`提出了`掩码交叉注意力`，并取代了MaskFormer中的交叉注意力。掩码交叉注意力使得对象查询只关注对象区域，由前面阶段的掩码输出引导。Mask2Former还采用了更强的可变形FPN主干、更强的数据增强和多尺度掩码解码。上述工作仅考虑更新对象查询。
> `CMT- Deeplab` 提出了对象查询和解码器特征的交替过程。它联合更新对象查询和像素特征。
> 之后，受k-means聚类算法的启发，`kMaX-DeepLab`通过在交叉注意力操作中引入聚类的argmax操作，提出了k-means交叉注意力
> PanopticSegformer  提出解耦查询策略和深度监督掩码解码器来加快训练过程。
> 
> SparseInst 提出一个稀疏的实例激活图集合，为每个前景对象突出信息区域。

> 除了分割任务外，还有一些工作通过引入新的解码器设计来`加快DETR的收敛速度`，并且大多数方法可以扩展到IS中。
> 一些工作在DETR解码器中引入语义先验。
> SAM-DETR 将对象查询映射到语义空间，搜索具有最具判别性特征的显著点。
> SMAC   通过对high near estimated bounding box locations的采样特征进行位置感知协同注意力。
> 一些工作采用了动态特征reweight
> 从多尺度特征的角度
> AdaMixer   使用估计的偏移量对特征进行空间和尺度上的采样。它使用MLP对采样特征进行动态解码，构建了一个快速收敛的基于查询的检测器。
> ACT-DETR  使用局部敏感哈希对查询特征进行自适应聚类，并将查询关键字交互替换为原型关键字交互，以降低交叉注意力成本。
> 从特征重加权的角度
> Dynamic-DETR 使用基于感兴趣区动态卷积对DETR的编码器和解码器部分都引入了动态注意力。
> 受解码器特征稀疏性的启发，Sparse-DETR选择性地从解码器中更新参考令牌，并在编码器中对选择的令牌提出一个辅助检测损失以保持稀疏性。
> 综上所述，将特征动态分配到查询学习中，加快了DETR的收敛速度。
* **Spatial-Temporal Cross-Attention Design**
在扩展了视频域中的对象查询后，`每个对象查询表示一个跨不同帧的跟踪对象`
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307030008601.png)

> VisTR           一个对象查询可以直接输出时空掩码，而不需要额外的跟踪。
> TransVOD
> IFC
> TeViT
> Seqformer
> Mask2FormerVIS
#### 3.2.3 Optimizing Object Query
对象查询  
添加位置信息        提供了对查询特征进行采样的线索，以便更快地进行训练
采用额外的监督   重点是在DETR中除了默认的损失函数之外，设计特定的损失函数

* **Adding Position Information into Query**
> 
* 