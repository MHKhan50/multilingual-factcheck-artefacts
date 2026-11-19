# Data sources and acquisition plan

This v0.1.0 release ships only synthetic fixture text. It does not bundle scraped news articles or redistribute third-party datasets.

Potential research datasets:
1. **RU22Fact**: multilingual Russian-Ukrainian war fact-checking dataset. Inspect the paper and dataset license/terms before downloading or redistribution.
   - Paper: https://aclanthology.org/2024.lrec-main.1245/
2. **SWARM**: multilingual dataset for studying information operations and narratives in the Russia-Ukraine war. Inspect annotation scope and license before use.
   - Dataset: https://huggingface.co/datasets/UKPLab/SWARM
3. **UCDP Georeferenced Event Dataset (GED)**: structured organised-violence event data, with codebooks and release-specific documentation.
   - Dataset portal: https://ucdp.uu.se/downloads/
   - API: https://ucdp.uu.se/apidocs/
4. **ACLED**: event data and conflict monitoring resources. Access, licensing, and redistribution may require registration or separate permission.
   - Data: https://acleddata.com/data-export-tool/
   - Methodology: https://acleddata.com/methodology/

Do not assume that different providers' event definitions, coverage, casualty thresholds, or inclusion criteria are interchangeable. Record dataset version, retrieval date, license, and checksum in a source manifest. Never redistribute data unless its terms permit it.
