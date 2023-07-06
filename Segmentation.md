https://github.com/hhaAndroid/awesome-mm-chat/blob/main/visual_segmentation.md

## 1. HIPIE: Hierarchical Open-vocabulary Universal Image Segmentation

https://github.com/berkeley-hipie/HIPIE
整体思路就是将things与stuff解耦，包括与文本特征在不同阶段做融合，分别使用不同的head做输出。
### Abstract
复杂的视觉场景可以被自然地分解为更简单的部分，并在多个粒度层次上进行抽象，从而引入固有的分割歧义。我们的方法积极地将包含不同语义层次的分层表示融入到学习过程中。针对"things"和"stuff"提出一种解耦的文本-图像融合机制和表示学习模块。我们系统地考察了这些类别之间在文本和视觉特征上存在的差异。
统一框架下HIerarchical, oPen-vocabulary, unIvErsal分割任务
超过40个数据集 
ADE20K
COCO
Pascal-VOC Part
RefCOCO、RefCOCOg
ODinW
SeginW
HIPIE
语义级分割、实例级分割(e.g., instance segmentation and referring segmentation)、部件级分割
![Pasted image 20230706165846|625](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061659861.png)
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061733990.png)

### Introduction
分析stuff和thing类的视觉和文本相似度矩阵，得到如下结论：
1. stuff类和thing类的文本特征和视I觉特征的类间相似性存在明显差异。
2. 与things相比，stuff类在文本特征上表现出显著更高的相似度
![image.png|400](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061707925.png)
这一观察表明，与stuff类相比，整合文本特征在为things类生成判别性特征方面可能会产生更显著的好处。
对于thing classes，使用早期的图像-文本融合方法，以充分利用具有判别性的文本特征的优势。
对于stuff classes， 我们使用一种后期的图像-文本融合策略来缓解非判别性文本特征引入的潜在负面影响。
此外，stuff类和thing类在视觉和文本特征上的差异，以及它们在特征(stuff类需要更好地捕捉纹理和材料,而thing类往往具有良好定义的几何形状,需要更好地捕捉形状信息)上的固有差异，表明需要**解耦表示学习模块**来为物品和事物生成掩码。
> 原始统一的表示学习模块-------> 对于thing和stuff classes解耦的表示学习模块
### Method
#### 3.1 整体框架
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061748908.png)
三个主要部件：
文本-图像特征提取和信息融合：
前景(things)和背景(stuffs)掩码生成：多个head， 返回(Mask、Box、Embedding)
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061757069.png)
利用文本提示进行提议和掩膜检索
#### 3.2 Text Prompts
#### 3.3 Image and Text Feature Extraction
BERT 512 tokens, 将更长的序列划分为512个令牌的片段，并对每个片段进行单独编码, 然后将得到的特征进行拼接，得到原始序列长度的特征
ResNet-50 最后3个blocks
ViT    blocks 8， 16， 32
#### 3.4 Text-Image Feature Fusion
双向交叉注意力(Bi-Xattn)提取文本引导的视觉特征$F_{t2v}$和图像引导的文本特征$F_{v2t}$，这些注意力特征再通过残差连接与原始的文本特征$F_t$和图像特征$F_v$集成，如下所示：
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061855511.png)
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061856564.png)
#### 3.5 Thing and Stuff Mask 生成
* **Model Architecture**: 
> Mask2Former and MaskDINO 与针对特定任务(例如,仅实例分割)训练的相同模型相比，联合训练的模型表现出更差的性能。我们假设这可能是**由于前景实例掩码和背景语义掩码的空间位置和几何形状分布不同造成的**。
> -----> 采用独立解码器
> 对于thing decoder，我们采用了Deformable DETR ，其掩模头遵循UNINEXT架构，并融入了DINO提出的去噪步骤。
> 对于stuff decoder， 使用Mask DINO的架构

* **Proposal and Ground-Truth Matching Mechanisms**
> thing decoder 采用**simOTA** 来执行框提议和背景真值之间的多对一匹配, 还使用了基于box - iou的NMS来去除重复的预测。
> stuff decoder  匈牙利一对一匹配
> 对于stuff masks禁用box loss.
> things quries: 900     stuffs quries: 300

* **Loss Functions**
> ![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061910007.png)
> 注意，虽然我们没有使用stuff解码器进行thing预测，但我们仍然将其预测与thing进行匹配，并计算训练中的类别和框损失。我们发现这样的辅助损失设置使得stuff解码器意识到了thing的分布并提高了最终的性能。
#### 3.6 Open-Vocabulary Universal Segmentation
> 特征E+CLIP特征$E_{CLIP}$
> $p_1(E, C_{test}) = \mathbb P(C_{test}|E)$
> $E_{CLIP} = MaskPooling(M, V(I))$
> $p_2(E, C_{test}) = \mathbb P(C_{test}|E_{CLIP})$
> ![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061925198.png)

#### 3.7 Hierarchical segmentation
> ![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061930507.png)
> 同时使用零件标签和实例标签对分类头进行监督
> ==**没有部件mask标签????**==
#### 3.8 Class-aware part segmentation with SAM
![|350](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307061946302.png)


## 2. Mask2Former
https://bowenc0221.github.io/mask2former
==masked attention==   将注意力限制在以预测片段为中心的局部特征上，这些特征可以是对象，也可以是区域
==multi-scale high-resolution features==
==optimization improvements== 
> 切换自注意力和交叉注意力的顺序
> 使查询特征具有可学习性
> 去除dropout
==calculating mask loss on few randomly sampled points==

### 3. Masked-attention Mask Transformer

#### 3.1 Mask classification preliminaries
	backbone
	pixel decoder
	Transformer decoder

#### 3.2 Transformer decoder with masked attention

![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307062311204.png)

##### 3.2.1 Masked attention
然而，最近的研究[ 22、46]表明，基于Transformer的模型的缓慢收敛是由于交叉注意力层中的全局上下文，因为交叉注意力需要许多训练时间来学习局部化的对象区域
--------------------------------------------------->
我们假设局部特征足以更新查询特征，上下文信息可以通过自注意力收集
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307062322716.png)
送入decoder前利用$X_0$阈值0.5得到一个二值预测$M_0$

##### 3.2.2 High-resolution features
不总是使用高分辨率的特征图，使用由低分辨率和高分辨率特征组成的特征金字塔，每次将多尺度特征的一个分辨率输入到Transformer解码器层。

加position embedding(正弦)和level embedding(可学习)

##### 3.2.3 Optimization improvements
	标准Transformer Decoder：自注意力模块、交叉注意力和前馈网络( FFN )。
	查询特征($X_0$)在输入Transformer解码器之前被初始化为零，并与可学习的位置嵌入相关
* 切换自注意力和交叉注意力的顺序
* 使查询特征具有可学习性
> ==在Transformer解码器中使用可学习的查询特征$X_0$来预测掩码( M0 )之前，直接对其进行监督
> （加额外损失，用标签监督？？？）
* 去除dropout

#### 3.3. Improving training efficiency
更具体地说，在构造用于二部匹配的代价矩阵的匹配损失中，我们对所有预测和真实掩码统一采样同一组K点。
在预测和其匹配的真值之间的最终损失中，我们使用`importance sampling`为不同的预测和真值对采样不同的K点集。
K = 12244 = 112x112


## 3. X-Detector


## 4. Universal Instance Perception as Object Discovery and Retrieval
https://github.com/MasterBin-IIAU/UNINEXT
