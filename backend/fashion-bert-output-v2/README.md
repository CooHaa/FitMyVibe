---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:149
- loss:LoggingMultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Soft materials and darker tones for a quiet confidence.
  sentences:
  - Vintage Realtree Camo Zip-Up Jacket Size Large Tall Excellent Condition   vintage
    camo y2k outdoors streetwear
  - Zara leather washed faux leather overshirt. Relaxed fit overshirt made of faux
    leather. Lapel collar and long sleeves with buttoned cuffs. Chest patch pocket.
    Washed effect. Front snap button closure.
  - Zara blue / black washed puffer jacket. Relaxed fit jacket made of cotton fabric
    with padded interior. Lapel collar and long sleeves with buttoned cuffs. Zip pockets
    at chest and hip. Washed effect. Front zip closure.
- source_sentence: A flowy white outfit that feels light and dreamy.
  sentences:
  - Zara sage green textured knit polo. Regular fit knit polo shirt made with lyocell
    and linen blend yarn. Lapel collar with front opening. Short sleeves. Ribbed trim.
  - Zara white fitted shirt zw collection. zara woman collection  T-shirt made of
    100% cotton yarn. Notched lapel collar and short sleeves. Pleated detail at the
    waist. Front button closure.
  - New with tags Hot Pink & cream colored Tall boot socks
- source_sentence: Something utilitarian that still feels considered.
  sentences:
  - The North Face Tan and cream chino pants. They are straight leg so they have a
    great baggy look to them. Make this your choice for you next amazing outfit idea.
    Don’t miss out on a great price         streetwear thrifted pants menfashion pants
  - Blue and yellow vintage 90s crewneck sweatshirt Russell size large    vintage
    russell crewneck company
  - Vintage Polo Ralph Lauren long sleeve shirt. Nice worn red color, relaxed fit.
    Measurements in pictures  streetwear preppy vintage worn faded
- source_sentence: Do you have any graphic tees that feel nostalgic and fun?
  sentences:
  - 'Zara orange nylon crossbody bag. Crossbody bag made of technical fabric. Main
    compartment with zip closure. Has a flat interior pocket with zip closure. Front
    flap pocket and quick-release buckle closure. Removable coin purse. Removable
    and adjustable shoulder strap.   Height x Length x Width: 5.5 x 7.9 x 2.4 inches
    (14 x 20 x 6 cm)'
  - 'lanascloset ~~~ description: never worn! i delete/update my listings and relist
    them so like my “sold” listings to have easier access to my shop later on ~~~
    i normally ship the following day, but it happens that i ship a few days after
    purchase ~~~ forever 21 brandy Melville baseball tee'
  - 3 Pack Vintage Mystery Shirt Bundle |  $15 spring special   ~ just buy your size
    (will fit to that size regardless of tag)  ~ vintage, Y2K, sports  vintage y2k
    mystery bundle
- source_sentence: A polished but easygoing outfit I can wear to work and dinner.
  sentences:
  - Zara gray easy care jogger waist pants. Pants with adjustable elastic drawstring
    waistband. Side pockets and back welt pockets. Cuffed hem.
  - Zara brown straight leg drawstring pants. Mid-rise pants with elastic waistband
    and adjustable drawstring. Side pockets and back patch pockets.
  - Zara oyster-white straight leg drawstring pants. Mid-rise pants with elastic waistband
    and adjustable drawstring. Side pockets and back patch pockets.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
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
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
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
    'A polished but easygoing outfit I can wear to work and dinner.',
    'Zara gray easy care jogger waist pants. Pants with adjustable elastic drawstring waistband. Side pockets and back welt pockets. Cuffed hem.',
    'Zara oyster-white straight leg drawstring pants. Mid-rise pants with elastic waistband and adjustable drawstring. Side pockets and back patch pockets.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

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
  | details | <ul><li>min: 5 tokens</li><li>mean: 12.7 tokens</li><li>max: 19 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 29.89 tokens</li><li>max: 86 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 32.28 tokens</li><li>max: 166 tokens</li></ul> |
* Samples:
  | sentence_0                                                     | sentence_1                                                                                                                                                                                         | sentence_2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
  |:---------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>lightweight shirt for summer</code>                      | <code>Zara beige lightweight faux suede overshirt. Relaxed fit overshirt made of a lightweight, faux suede fabric. Lapel collar and long sleeves with buttoned cuffs. Front button closure.</code> | <code>Victoria secret Pink yogi leggings. They are made of a athletic material and are Not cotton. They are Not see through. They have a maroon logo band. They are a size large and are full length. They are in great condition. Comes from a smoke and pet free home. ****The band is maroon**** Faq: Can I bundle? Yes and when you bundle you get a discount for Any of my items. I ship very fast How fast is shipping? I usually Goto the post office every other day except for Sunday's. I use usps which has very fast shipping. Are there returns? Yes you can return your item if it is not the item I described and in the photos.</code> |
  | <code>A leather texture that adds instant cool.</code>         | <code>Brand new tillys black tube top  Size M</code>                                                                                                                                               | <code>Super classy top, size is flexible but is marked as xxl</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  | <code>An outfit with modern lines but a nostalgic edge.</code> | <code>Zara black flowy pleated pants. Mid-rise pants with interior elastic waistband. Front pleats detail. Wide leg.</code>                                                                        | <code>Ugg Bailey bow boots size 7 Still has a lot of life in them The bottom sole of the boot is a little worn and the boots has some lite stains please look at the pictures</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
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