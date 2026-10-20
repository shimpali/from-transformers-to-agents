# Image captions (paste under each image on Medium)

- `01-pipeline-overview.svg`: Run this once per output token, feeding each new token back in, and you get generation.
- `02-encoder-vs-decoder.svg`: GPT-style LLMs stack only decoder blocks, minus the dashed cross-attention layer: there's no encoder output left to attend to.
- `03-tokenization-example.svg`: The model never sees the string, only this list of integer IDs.
- `04-embeddings-2d-space.svg`: Similar meanings sit close together, and consistent relationships (male → female) point the same way.
- `05-self-attention-example.svg`: Most of it lands on “cat”, and that’s how the model resolves the pronoun.
- `06-transformer-block.svg`: Dashed lines are residual connections: they carry the input forward unchanged, so a block only has to learn what to add, not the whole representation from scratch.
- `07-generation-loop.svg`: Every loop produces exactly one new token, using the whole sequence generated so far. Repeat until an end-of-sequence token appears or a length limit is hit.
