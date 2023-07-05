

### 1. 环境配置

coco 数据集下载

```shell
python tools/misc/download_dataset.py --dataset-name coco2017
```

```shell
# 将全景分割格式转成实例分割（评估使用）
python datasets/panoptic2detection_coco_format.py --things_only
```

下载了coco/150_16_swin_l_oneformer_coco_100ep.pth官方训练好的模型推理，精度基本一致

|  Method   | Backbone |  PQ  | PQTh |   PQSt   |  AP  | mIoU | #params |                            config                            |                          Checkpoint                          |
| :-------: | :------: | :--: | :--: | :------: | :--: | :--: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| OneFormer | Swin-L†  | 57.9 | 64.4 |   48.0   | 49.0 | 67.4 |  219M   | [config](./configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml) | [model](https://shi-labs.com/projects/oneformer/coco/150_16_swin_l_oneformer_coco_100ep.pth) |
|           |          | 57.9 | 64.4 |   48.0   | 49.0 | 67.2 |         |                                                              |                                                              |
| OneFormer | DiNAT-L† | 58.0 | 64.3 |   48.4   | 49.2 | 68.1 |  223M   | [config](https://github.com/SHI-Labs/OneFormer/blob/main/configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml) | [model](https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth) |
|           |          | 58.0 | 64.3 | **48.3** | 49.2 | 68.1 |         |                                                              |                                                              |

### 2. 数据加载



#### 2.1 DatasetMapper

![image-20230703185229289](https://huyanchen-1315211807.cos.ap-beijing.myqcloud.com/images/image-20230703185229289.png)

统一Panoptic标签读取，训练时随机处理成语义分割、实例分割、全景分割

```python
def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None) #推理无需标签
            return dataset_dict

        # semantic segmentation
        # 读取语义分割标签
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
        else:
            sem_seg_gt = None
        # 读取全景分割标签用于后续获取instances
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id
            pan_seg_gt = rgb2id(pan_seg_gt)

        prob_task = np.random.uniform(0,1.) # 生成0， 1区间随机数

        num_class_obj = {}

        for name in self.class_names:
            num_class_obj[name] = 0
		
        # self.semantic_prob = 0.33, self.instance_prob=0.66,随机处理成语义分割、实例分割、全景分割任务形式输入
        if prob_task < self.semantic_prob:
            task = "The task is semantic"
            instances, text, sem_seg = self._get_semantic_dict(pan_seg_gt, image_shape, segments_info, num_class_obj)
        elif prob_task < self.instance_prob:
            task = "The task is instance"
            instances, text, sem_seg = self._get_instance_dict(pan_seg_gt, image_shape, segments_info, num_class_obj)
        else:
            task = "The task is panoptic"
            instances, text, sem_seg = self._get_panoptic_dict(pan_seg_gt, image_shape, segments_info, num_class_obj)


        dataset_dict["sem_seg"] = torch.from_numpy(sem_seg).long() #(h, w) map
        dataset_dict["instances"] = instances     #['gt_masks', 'gt_boxes']
        dataset_dict["orig_shape"] = image_shape  #
        dataset_dict["task"] = task               #"The task is"
        dataset_dict["text"] = text               #
        dataset_dict["thing_ids"] = self.things   

        return dataset_dict
```

```python
def _get_semantic_dict(self, pan_seg_gt, image_shape, segments_info, num_class_obj):
        instances = Instances(image_shape)
        
        classes = []
        #self.num_queries = cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES cfg.MODEL.TEXT_ENCODER.N_CTX
        texts = ["a semantic photo"] * self.num_queries  
        masks = []
        label = np.ones_like(pan_seg_gt) * self.ignore_label

        # 处理成语义分割标签， class不存在就添加该mask，class存在就加到masks[idx]中
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                mask = pan_seg_gt == segment_info["id"]
                if not np.all(mask == False):
                    if class_id not in classes:
                        cls_name = self.class_names[class_id]
                        classes.append(class_id)
                        masks.append(mask)
                        num_class_obj[cls_name] += 1
                    else:
                        idx = classes.index(class_id)
                        masks[idx] += mask
                        masks[idx] = np.clip(masks[idx], 0, 1).astype(np.bool) # 限制在0，1之间
                    label[mask] = class_id 

        # 根据instances修改texts
        num = 0
        for i, cls_name in enumerate(self.class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1
                    
        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_bboxes = torch.zeros((0, 4))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            # Placeholder bounding boxes for stuff regions. Note that these are not used during training.
            instances.gt_bboxes = torch.stack([torch.tensor([0., 0., 1., 1.])] * instances.gt_masks.shape[0]) # stuff box标签[0, 0, 1, 1]
        return instances, texts, label #(h, w)
```

#### 2.2 数据增强

OneFormerUnifiedDatasetMapper与COCOUnifiedNewBaselineDatasetMapper数据增强不同

```python
# OneFormer
augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
           #带单类面积比重约束的随机位置图像裁剪，仅用于实例分割。类似于RandomCrop，但引入单类面积比重上界，判断随机裁剪位置中各类分割掩码面积比重，当某类比重越过上界时，重新随机（当仅含一类时，也重新随机；可设置忽略某类）。以上过程最多进行10次，仍未满足时采用第十次裁剪结果
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
            
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())
```

```python
# coco
def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE # 1024
    min_scale = cfg.INPUT.MIN_SCALE # 0.1
    max_scale = cfg.INPUT.MAX_SCALE # 2.0

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation
```

### 3. model

#### 3.1 meta-arch

![image-20230703225909093](C:\Users\HuYanchen\AppData\Roaming\Typora\typora-user-images\image-20230703225909093.png)

整体结构部分，主要是对新加入的text和task的处理

```python
def forward(self, batched_inputs):
		#对任务文本（The task is {}）进行序列化，经过MLP
        tasks = torch.cat([self.task_tokenizer(x["task"]).to(self.device).unsqueeze(0) for x in batched_inputs], dim=0) #(b, 77)
        tasks = self.task_mlp(tasks.float()) #(b, 256)

        features = self.backbone(images.tensor) 
        outputs = self.sem_seg_head(features, tasks)

        if self.training:
            #对texts进行稀疏化，送入编码器
            texts = torch.cat([self.text_tokenizer(x["text"]).to(self.device).unsqueeze(0) for x in batched_inputs], dim=0) #b, n, l
            texts_x = self.encode_text(texts)

            outputs = {**outputs, **texts_x} # 利用text_x计算对比损失
```

#### 3.3 Text_encoder

```python
def encode_text(self, text):
        assert text.ndim in [2, 3], text.ndim
        b = text.shape[0]
        squeeze_dim = False
        num_text = 1
        if text.ndim == 3:
            num_text = text.shape[1]
            text = rearrange(text, 'b n l -> (b n) l', n=num_text)
            squeeze_dim = True

        x = self.text_encoder(text)

        text_x = self.text_projector(x) #MLPs

        if squeeze_dim:
            text_x = rearrange(text_x, '(b n) c -> b n c', n=num_text)
            if self.prompt_ctx is not None:
                text_ctx = self.prompt_ctx.weight.unsqueeze(0).repeat(text_x.shape[0], 1, 1) # B, 16, D
                text_x = torch.cat([text_x, text_ctx], dim=1)# 这里cat了一个Q_ctx
        
        return {"texts": text_x}
```

![image-20230704023427526](C:\Users\HuYanchen\AppData\Roaming\Typora\typora-user-images\image-20230704023427526.png)    



Text_encoder

每句话L个词，对每个位置每句话的词做交互

```python
def forward(self, text): # B*N，L
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] # N，D 取最后一个词
        return x
```

#### 3.2 TransformerDecoder

![image-20230704062728423](C:\Users\HuYanchen\AppData\Roaming\Typora\typora-user-images\image-20230704062728423.png)

在mask2former基础上加入了task预测

```python
		query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) #Nq,b,d
        tasks = tasks.unsqueeze(0)  #1,b,d
        if self.use_task_norm:
            tasks = self.decoder_norm(tasks)
        
        feats = self.pe_layer(mask_features, None) #只有位置编码？

        out_t, _ = self.class_transformer(feats, None, 
                                    self.query_embed.weight[:-1],#Nq-1, d 
                                    self.class_input_proj(mask_features),  #1x1卷积 map
                                    tasks if self.use_task_norm else None)
  
        out_t = out_t[0].permute(1, 0, 2)
        #1,b,Nq,d -------->b,Nq,d--------->Nq,b,d
        out = torch.cat([out_t, tasks], dim=0)
```

```python
def forward(self, src, mask, query_embed, pos_embed, task_token=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)#b,d,h,w-->b,d,h*w-->N,b,d
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)#Nq-1,b,d
        if mask is not None:
            mask = mask.flatten(1)
            
        if task_token is None:
            tgt = torch.zeros_like(query_embed)
        else:
            tgt = task_token.repeat(query_embed.shape[0], 1, 1)#Nq-1,b,d
   
        #encoder q,k+pos v没加
    	memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)#H*W,b,d
        hs = self.decoder(
            tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed
        )#1,Nq-1,b,d 这里decoder输出unsqueeze(0)否则维度对不上
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
```

