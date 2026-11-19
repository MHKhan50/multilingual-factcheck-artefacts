# Data sources

## Used in v0.2.0: RU22Fact

- **What:** 16,033 fact-checked claims about the Russia-Ukraine conflict in English, Ukrainian, Russian
  and Chinese. Each claim has LLM-optimised evidence, a reference explanation and a label
  (Supported, Refuted, NEI), with train, validation and test splits.
- **Paper:** Zeng et al. (2024), *RU22Fact: Optimizing Evidence for Multilingual Explainable Fact-Checking
  on Russia-Ukraine Conflict*, LREC-COLING 2024, pp. 14215-14226,
  <https://aclanthology.org/2024.lrec-main.1239>.
- **Repository:** <https://github.com/zeng-yirong/ru22fact>, pinned to commit
  `c467abb1c98bd4fc043ac0fd5a03c4f1afa109d0`.
- **Licence:** the repository has no licence file. This project therefore **does not redistribute**
  the data. `civintel ru22fact-download` fetches the three CSV files from the pinned commit for
  research use and checks their SHA-256 digests (listed in `src/civintel_agentbench/ru22fact.py`).
- **Known issues:** two rows (one in train, one in test) have shifted columns and no language;
  they are dropped and counted in each run manifest. Some claims come from general fact-checking
  feeds and are not about the conflict. The evidence was written by an LLM and may contain errors;
  it is treated as model input, not as fact.

## Considered and not used

1. **UCDP Georeferenced Event Dataset (GED):** structured event records without report text;
   <https://ucdp.uu.se/downloads/>.
2. **ACLED:** event data that require registration and have redistribution terms;
   <https://acleddata.com/data-export-tool/>.
3. **DNIPRO** (Mohanty et al., 2026): 246,229 news articles in English, Russian and Chinese under
   CC BY-NC 4.0, released as metadata and annotations only; the article text must be fetched from the
   original URLs. <https://zenodo.org/records/18470677>.
4. **SWARM** (arXiv:2609.12653): Russian propaganda in search-engine results, gated and research-use
   only; a different task. <https://huggingface.co/datasets/manueltonneau/SWARM>.

Record dataset version, retrieval date, licence and checksum for any new source, and never
redistribute data unless its terms permit it.
