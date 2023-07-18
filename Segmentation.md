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

### 2.3. Masked-attention Mask Transformer

#### 2.3.1 Mask classification preliminaries
	backbone
	pixel decoder
	Transformer decoder

#### 2.3.2 Transformer decoder with masked attention

![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307062311204.png)

##### 2.3.2.1 Masked attention
然而，最近的研究[ 22、46]表明，基于Transformer的模型的缓慢收敛是由于交叉注意力层中的全局上下文，因为交叉注意力需要许多训练时间来学习局部化的对象区域
--------------------------------------------------->
我们假设局部特征足以更新查询特征，上下文信息可以通过自注意力收集
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307062322716.png)
送入decoder前利用$X_0$阈值0.5得到一个二值预测$M_0$

##### 2.3.2.2 High-resolution features
不总是使用高分辨率的特征图，使用由低分辨率和高分辨率特征组成的特征金字塔，每次将多尺度特征的一个分辨率输入到Transformer解码器层。

加position embedding(正弦)和level embedding(可学习)

##### 2.3.2.3 Optimization improvements
	标准Transformer Decoder：自注意力模块、交叉注意力和前馈网络( FFN )。
	查询特征($X_0$)在输入Transformer解码器之前被初始化为零，并与可学习的位置嵌入相关
* 切换自注意力和交叉注意力的顺序
* 使查询特征具有可学习性
> ==在Transformer解码器中使用可学习的查询特征$X_0$来预测掩码( M0 )之前，直接对其进行监督
> （加额外损失，用标签监督？？？）
* 去除dropout

#### 2.3.3. Improving training efficiency
更具体地说，在构造用于二部匹配的代价矩阵的匹配损失中，我们对所有预测和真实掩码统一采样同一组K点。
在预测和其匹配的真值之间的最终损失中，我们使用`importance sampling`为不同的预测和真值对采样不同的K点集。
K = 12244 = 112x112


## 3. X-Decoder
Generalized Decoding for Pixel, Image, and Language
https://x-decoder-vl.github.io
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/20230710031145.png)
贡献：
* 通用的解码器框架
> 两组查询作为输入: `通用的非语义查询`、`新引入的文本查询`
> 预测两种类型的输出：`pixel-level masks`、`token-level semantics`
> 使用单个文本编码器对文本语料进行编码
* 端到端的学习范式
> 从所有粒度的监督中学习
> 将三种类型的数据统一起来：全景分割、指称分割和图文对。
* 对大范围的分割和VL任务具有较强的零样本和任务相关的迁移能力。
![image.png|900](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/20230710034159.png)
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/20230710035726.png)

## 4. UNINEXT：Universal Instance Perception as Object Discovery and Retrieval
https://github.com/MasterBin-IIAU/UNINEXT
十个object-centric任务，按照形式-时序-参考归类      instance perception
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/20230709210622.png)
下一代通用实例感知模型
根据提示不同将感知任务分成三组：
* 类别名称作为提示：(Object Detection, Instance Segmentation, VIS, MOT, MOTS)
* 语言描述作为提示：(REC, RES, R-VOS)
* 参考注释作为提示：(SOT, VOS)
prompt-guided object discovery and retrieval formulation
首先在提示符的引导下发现N个对象提议，然后根据实例-提示符匹配得分从提议中检索出最终实例
`a prompt generation module`包含参考文本编码器和参考视觉编码器
`early fusion module`：增强当前图像视觉特征和prompt embeddings
`instance decoder`:  Deformbale DETR
主要贡献：
* 提出了一个统一的提示引导的通用实例感知方案，将以前碎片化的实例级子任务重新组合成一个整体。
* 得益于灵活的对象发现和检索范式，UNINEXT可以在不同的任务和领域上进行训练，不需要特定的任务头。
* UNINEXT在10个实例感知任务的20个具有挑战性的基准测试集上，使用相同模型参数的单一模型取得了优越的性能。
#### 4.2 Related Work
GLIP+Unicorn
#### 4.3 Method
![image.png|825](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/20230709221021.png)
`prompt generation`、`image-prompt feature fusion` 、`object discovery and retrieval`.
##### 4.3.1 Prompt Generation
对于language-related prompts：
>采用文本编码器$Enc_{L}$，对于类别引导任务，将类别名串联起来作为语言描述。eg: For coco
 `person. bicycle. ... . toothbrush`

对于annotation-guided任务：
> 采用额外的reference visual encoder $Enc_V^{ref}$
> 目标中心扩大$2^2$倍 ----> 256x256 ----> 串接一个target prior通道(类似mask) -----> $Enc_{V}^{ref}$
> ${C_3, C_4, C_5, C_6}   ---->   32 \times 32, 16\times16, 8\times8, 4\times4$
> 引入merging module（保留精细的目标信息，并得到与其他任务相同格式的提示嵌入）上采样到$32 \times 32$然后add
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/20230710000848.png)
##### 4.3.2 Image-Prompt Feature Fusion
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/20230710004336.png)
采用6个视觉-语言融合层和6个额外的BERT层进行特征增强
##### 4.3.2 Object Discovery and Retrieval
Deformable DETR  head
* 不随图像或提示变化的静态查询
* 以提示为条件的动态查询
> 在序列维度上对$F_v'$做池化，重复N次
> 静态查询通常比动态查询具有更好的性能

预测头引入embedding head与先前轨迹做关联，用在MOT、MOTS、VIS任务

对proposal做检索：
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/20230710021316.png)
$$S=F_{ins}W^T$$

##### 4.3.4 Training and Inference
**训练**（三个连续阶段）
* 一般感知预训练  
> Objects365        
> 无mask注释，引入==BoxInst==提出的两个辅助损失训练mask分支
> $$L_{stage1} = L_{retrieve} + L_{bbox} + L_{mask}^{boxinst}$$
> 采用Focal loss监督
* 图像级联合训练
> 进一步微调
> COCO、RefCOCO、RefCOCO+、RefCOCOg
> mask分支：Dice loss、Focal loss
> $$L_{stage2} =  L_{retrieve} + L_{bbox} + L_{mask}$$
* 视频级联合训练
> video-level datasets
> 在这一阶段，模型从原始视频中随机选取两帧进行训练。为了避免模型遗忘先前学习到的关于图像级别任务的知识，我们还将图像级别的数据集转化为伪视频与其他视频数据集进行联合训练。
> COCO、RefCOCO/g/+
> GOT-10K , LaSOT , TrackingNet , and Youtube-VOS
> BDD100K, VIS19, OVIS 
> Ref-Youtube-VOS
> 在这一阶段，加入了 `reference visual encoder` 和用于关联的`embedding head`
>  $$L_{stage3} =  L_{retrieve} + L_{bbox} + L_{mask}+L_{embed}$$

**推理**


## 5. Detecting Everything in the Open World: Towards Universal Object Detection
难点：
* 利用多源和异构标签空间训练 
> 现有的检测器只能从一个标签空间预测类别，并且数据集之间特定于数据集的分类和注释不一致使得难以统一多个异构标签空间
* 更好地推广到开放世界
> 新类歧视
> 在推理时，模型将偏向基类，并对新类产生信心不足的预测。尽管语言嵌入使得预测新类别成为可能，但它们的性能仍然远远低于基本类别

partitioned structure
利用region proposals新类的泛化能力，将region proposals生成阶段和 RoI 分类阶段解耦，而不是联合训练它们。
class-agnostic localization network (CLN)
概率校准来消除预测偏差
> 估计所有类别的先验概率，然后根据先验​​概率调整预测的类别分布

### 3.4 框架
![image.png|750](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307122303363.png)

* 大规模图像文本预训练   RegionCLIP
*  异构标签空间训练      解耦方式而不是之前的联合训练方法
* 开放世界推理          概率校准，保证新类别与旧类别推理平衡

#### 3.4.1 异构标签空间训练
![image.png](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307122309573.png)
分类头都采用区域特征和语言嵌入之间的相似性
单独的结构训练各个网络并将它们集成以进行推理
统一的结构将多个数据集统一为一个数据集
分区的结构共享相同的主干但不同的分类头
==为了避免类别数量增加时基于 sigmoid 的分类损失值过高，我们随机抽取一定数量的类别作为负类别。负类咋操作的呀???
* **Decoupling proposal generation and RoI classification**
RPN ImageNet初始化
RoI classification阶段用文本图像预训练参数
==两个backbone???==  两个模型？？
* **Class-agnostic localization network.**
![image.png|500](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/202307130002141.png)
box refinement
RPN定位置信度：$s_i^{r_1}$
RoI head定位置信度：$s_i^{r_2}$，分类置信度$s_i^{c}$（类别无关二分类）
总置信度$\eta _i = (s_i^c)^\alpha \cdot (s_i^{r_1}s_i^{r_2})^{(1-\alpha)}$

==定位置信度咋训练的???==    BoxIOU  L1Loss

#### 3.4.2 开放世界推理

概率校准
减少基类概率，增加新类概率
$$p_{ij} = \frac{1}{1+exp(-z_{ij}^Te_j/\tau)}/\pi_j^{\gamma}, j\in L_{test}$$
原始概率/类别$\pi_j$先验概率，
对测试数据推理，统计$\pi_j$
==实际应用中难以得到全部测试数据来统计==
$s_{ij} = p_{ij}^\beta \eta _i^{1-\beta}$



## 6. Unified Open-Vocabulary Dense Visual Prediction
## 7.  Semantic-SAM: Segment and Recognize Anything at Any Granularity
