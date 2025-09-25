set positional-arguments := true

prompt-rnn prompt="Transformers will" max_new_tokens="200" temperature="0.8":
    uv run python -m tinylm rnn prompt \
      --name rnn_full \
      --checkpoint best \
      --prompt {{quote(prompt)}} \
      --max-new-tokens {{max_new_tokens}} \
      --temperature {{temperature}}

prompt-transformer prompt="Attention is all you need" max_new_tokens="200" temperature="0.8":
    uv run python -m tinylm transformer prompt \
      --name tfm_full \
      --checkpoint best \
      --prompt {{quote(prompt)}} \
      --max-new-tokens {{max_new_tokens}} \
      --temperature {{temperature}}
