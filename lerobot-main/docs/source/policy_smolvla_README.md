## Vision prefix: `use_grid_token_sampler`

SmolVLA can compress image patch tokens with an `ActiveTokenSampler` before the VLM (`use_grid_token_sampler=true`, default) or use all patches (`false`). Configure it when training with `--policy.use_grid_token_sampler=...`; it is stored in `config.json`. The number of sampled vision tokens is `grid_token_sampler_num_tokens` (default `16`), overridable via `--policy.grid_token_sampler_num_tokens=N` for training; it must match the checkpoint when loading weights. At inference, `SmolVLAPolicy.forward`, `predict_action_chunk`, and `select_action` accept an optional `use_grid_token_sampler` keyword (`None` means use the config). Enabling the sampler at runtime requires that the policy was built with `use_grid_token_sampler=True` so weights are present.

## Paper

https://arxiv.org/abs/2506.01844

## Citation

```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```
