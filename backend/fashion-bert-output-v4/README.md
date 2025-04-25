---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:149
- loss:LoggingMultipleNegativesRankingLoss
base_model: sentence-transformers/msmarco-distilbert-base-dot-prod-v3
widget:
- source_sentence: A versatile layer that works in transitional seasons.
  sentences:
  - Zara black abstract print stretch shirt. Shirt made with a fabric that minimizes
    the need to iron after washing. Spread collar and short sleeves. Side vents at
    hem. Front button closure.
  - Vintage Harley Davidson Croatia rare shirt fits large  vintage harley harleydavidson
    croatia biker
  - Vintage Harley Davidson T Shirt  Adult L  Very good condition. No flaws.  HarleyDavidson
    Vintage Y2K Streetwear
- source_sentence: A quiet but confident piece with just enough shape.
  sentences:
  - Pink flower top w/ cowl neckline 💕  Such a cute feminine top great for summer.
    You can easily dress up or dress down this top for any occasion!
  - The North Face fleece - size L - fits true to size - excellent condition - no
    flaws
  - Size 10z men's. Open back slip on. Black fleece with flame and logo. Brand new
    in box.
- source_sentence: Something that balances playful and sophisticated energy.
  sentences:
  - new with tags/new in online packaging size large (two thermals) No Holds, No Tr-a-des.
    please don't ask me my lowest! i offer bundle deals! Pink victoria's secret NinasCloset
  - Vintage Realtree Camo Zip-Up Jacket Size Large Tall Excellent Condition   vintage
    camo y2k outdoors streetwear
  - Zara brown flowy wrap vest top. Flowy V-neck vest top. Adjustable tie front closure.
- source_sentence: comfortable pants for lounging
  sentences:
  - Zara black flowy pleated pants. Mid-rise pants with interior elastic waistband.
    Front pleats detail. Wide leg.
  - Men's Shirt ( size 54)
  - Zara oyster-white easy care jogger waist pants. Pants with adjustable elastic
    drawstring waistband. Side pockets and back welt pockets. Cuffed hem.
- source_sentence: An outfit that helps me feel grounded on anxious days.
  sentences:
  - Supreme mlk "Don’t Let the Dream Die" Tee -SS18 - White -  message before purchasing!  supreme
    streetwear skater t-shirt
  - Vintage Polo Ralph Lauren long sleeve shirt. Nice worn red color, relaxed fit.
    Measurements in pictures  streetwear preppy vintage worn faded
  - Hollister Strappy Sports Bra with optional removable pads. Size small recommended
    for 32c/34b/36a
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/msmarco-distilbert-base-dot-prod-v3

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/msmarco-distilbert-base-dot-prod-v3](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-dot-prod-v3). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/msmarco-distilbert-base-dot-prod-v3](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-dot-prod-v3) <!-- at revision 76ce77c5ed084214fd7319d3c92d75500c7c5220 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Dot Product
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: DistilBertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Dense({'in_features': 768, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'An outfit that helps me feel grounded on anxious days.',
    'Supreme mlk "Don’t Let the Dream Die" Tee -SS18 - White -  message before purchasing!  supreme streetwear skater t-shirt',
    'Hollister Strappy Sports Bra with optional removable pads. Size small recommended for 32c/34b/36a',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 149 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 149 samples:
  |         | sentence_0                                                                       | sentence_1                                                                        | sentence_2                                                                         |
  |:--------|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                            | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 12.7 tokens</li><li>max: 19 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 29.89 tokens</li><li>max: 86 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 30.96 tokens</li><li>max: 156 tokens</li></ul> |
* Samples:
  | sentence_0                                                                    | sentence_1                                                                                                                                                                                                                               | sentence_2                                                                                                                                                                                             |
  |:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Something sweet and vintage with just the right amount of flirt.</code> | <code>Vintage Fila Soccer graphic T-shirt  Size Large    vintage soccer streetwear fila skater</code>                                                                                                                                    | <code>Both have only been worn and washed once. The tags have been cut out tho.</code>                                                                                                                 |
  | <code>An outfit that says 'I make zines and drink herbal tea.'</code>         | <code>Pink flower top w/ cowl neckline 💕  Such a cute feminine top great for summer. You can easily dress up or dress down this top for any occasion!</code>                                                                             | <code>Zara black flowy wrap vest top. Flowy V-neck vest top. Adjustable tie front closure.</code>                                                                                                      |
  | <code>Rugged workwear jacket</code>                                           | <code>Zara black combination lapel jacket. Relaxed fit jacket made of lightly padded cotton fabric. Combination lapel collar and long sleeves with snap button cuffs. Patch pockets at the hip. Washed effect. Front zip closure.</code> | <code>Zara brick faux leather bomber jacket. Faux leather bomber jacket with lapel collar and long sleeves with elastic cuffs. Welt pockets on the front. Elastic hem. Front metal zip closure.</code> |
* Loss: <code>__main__.LoggingMultipleNegativesRankingLoss</code> with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 10
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.11.12
- Sentence Transformers: 3.4.1
- Transformers: 4.51.3
- PyTorch: 2.6.0+cu124
- Accelerate: 1.5.2
- Datasets: 3.5.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### LoggingMultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->