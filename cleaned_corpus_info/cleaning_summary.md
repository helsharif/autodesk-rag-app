# Corpus Cleaning Summary

Generated: 2026-05-21T10:45:04

## Overall Results

- Raw HTML files found: 1,218
- Initially cleaned before purge: 1,218
- Retained cleaned Markdown files after purge: 974
- Purged after cleaning: 244
- Cleaned file count reduction from purge: 20.03%
- Skipped: 0
- Failed: 0
- Very short cleaned documents: 345
- Total raw characters: 317,446,072
- Total cleaned characters: 3,574,578
- Approximate character reduction: 98.87%

## Status Counts

|         |   count |
|:--------|--------:|
| success |     974 |
| purged  |     244 |

## Extraction Method Counts

|                              |   count |
|:-----------------------------|--------:|
| trafilatura                  |     621 |
| trafilatura_short            |     311 |
| beautifulsoup_rich_fallback  |     135 |
| beautifulsoup_fallback       |      91 |
| beautifulsoup_table_fallback |      60 |

## Purge Policy

After cleaning and metadata enrichment, the pipeline deletes cleaned Markdown files smaller than 600 bytes. It also deletes cleaned Markdown files whose `document_language` header is a known non-English language. Documents with unknown language are retained.

## Largest Character Reductions

| relative_path                              |   char_count_before_cleaning |   char_count_after_cleaning |   char_reduction |   reduction_pct | extraction_method_used   |
|:-------------------------------------------|-----------------------------:|----------------------------:|-----------------:|----------------:|:-------------------------|
| adsk-130c9df036aa42d6912c4d1f1bc4048e.html |                      3218883 |                       13715 |          3205168 |           99.57 | trafilatura              |
| adsk-7296ab22bc024c25a8bf012b2db5a443.html |                      3186571 |                        6373 |          3180198 |           99.8  | trafilatura              |
| adsk-c0f3dd3a430a4db9a29b36d53ddf70a3.html |                      3156741 |                        4313 |          3152428 |           99.86 | trafilatura              |
| adsk-c247f0e7fa584e39a059b6b229ae0a45.html |                      3159198 |                        9603 |          3149595 |           99.7  | trafilatura              |
| adsk-71203d4e58e94ccda9622b0fe00de5fd.html |                      3125422 |                       14041 |          3111381 |           99.55 | trafilatura              |
| adsk-beaf70cfd5c841e2bdd0566e0dbcdc0c.html |                      3115100 |                        9162 |          3105938 |           99.71 | trafilatura              |
| adsk-890d3910f5954b14bfda6978a9527e9e.html |                      3115041 |                        9360 |          3105681 |           99.7  | trafilatura              |
| adsk-2348b52e0184498b97c24a7bba7c339a.html |                      3110518 |                        5396 |          3105122 |           99.83 | trafilatura              |
| adsk-6e42a9e81abc472bb380cd2f211afc7e.html |                      3109220 |                        4425 |          3104795 |           99.86 | trafilatura              |
| adsk-fe49a4714ee44c29b1797e8d42e73785.html |                      3109376 |                        5345 |          3104031 |           99.83 | trafilatura              |
| adsk-808e854f3670402195c6638b28fcf818.html |                      3114153 |                       11819 |          3102334 |           99.62 | trafilatura              |
| adsk-34f0319fec4c49d69821767b8c38e8ce.html |                      3103871 |                        6536 |          3097335 |           99.79 | trafilatura              |
| adsk-ba0cb39ac00d4c22903744baf35e41d4.html |                      2442623 |                        2104 |          2440519 |           99.91 | trafilatura              |
| adsk-7d32416bfef6465488086e31450c2e64.html |                      1986841 |                        1835 |          1985006 |           99.91 | trafilatura              |
| adsk-a6a5e4cfc66f4370baf1e2348a0981de.html |                      1925976 |                        1949 |          1924027 |           99.9  | trafilatura              |
| adsk-60b5544cccf24560a73a87902cc564d2.html |                      1908162 |                        1835 |          1906327 |           99.9  | trafilatura              |
| adsk-78c3e69579864d99b2d41bfc7043eed7.html |                      1834361 |                        1817 |          1832544 |           99.9  | trafilatura              |
| adsk-0ed32d8dbdfc4a17b269cf489b007f30.html |                      1828717 |                         101 |          1828616 |           99.99 | trafilatura_short        |
| adsk-a0e50a4d29b94fef8dd716eee7abc9c1.html |                      1828717 |                         101 |          1828616 |           99.99 | trafilatura_short        |
| adsk-45dc94ca846d4577be9e5d0886420172.html |                      1828717 |                         101 |          1828616 |           99.99 | trafilatura_short        |

## Shortest Cleaned Outputs

| relative_path                              | title                                   |   char_count_after_cleaning | extraction_method_used   | warnings                                                  |
|:-------------------------------------------|:----------------------------------------|----------------------------:|:-------------------------|:----------------------------------------------------------|
| adsk-19696090678a46c4aeee13c626b78358.html | Generative design Archives              |                          28 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-bde74d738109446ab3420848b73cebf3.html | Smart manufacturing Archives            |                          30 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-9d62b8985ddf44baa34e03495eea8e46.html | Design Visualization Archives           |                          31 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-d166a5a1ed7d4b3b833d5264f5e1eaa5.html | Custom Manufacturing Archives           |                          31 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-306d35ca49a0472e9300cf72c6f88bda.html | How to Renew your Premium plan          |                          32 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-0966028aeb674e83add24a3ba3a4361d.html | Sustainability in AEC Archives          |                          32 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-606ecba32b9b46c090392148a6d9784b.html | Media & Entertainment Archives          |                          32 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-5457db7805d84d72a953364fca41f9c0.html | Autodesk Moldflow Certification         |                          33 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-fb3a6e96a9ab4644a149a5a3712374b0.html | Sustainability in PD&M Archives         |                          33 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-ca523875e52b48529046a4c7089b28ec.html | Flame Family system requirements        |                          34 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-21c6e6cea84a4b7c8a3d103e4ee58703.html | What is the future of SketchBook?       |                          35 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-098e9a351e524634b450862861140de3.html | System requirements for Autodesk Maya   |                          39 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-f0702460015846dda0f1065e8e2be249.html | Inventor Engineer-to-Order | Autodesk   |                          39 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-61b6c979d84c496ab0ff25deac4be1a4.html | Autodesk Forma Learning Hub & Support   |                          39 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-ea2ae6835a9c4dfc8e19cf41f1b06ce5.html | System Requirements for Autodesk Forma  |                          40 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-7e02ac69f1d54ed4ab14f95a359fed52.html | Genuine Autodesk | Report nonvalid use  |                          40 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-2fcb34e959814b48b2fbf7c85029b872.html | ReCap Photo Frequently Asked Questions  |                          40 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-3f7e75aa6ef747bab1339841016864e1.html | System requirements for Autodesk EAGLE  |                          40 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-d0fe77933b1249d5a6f4bf34983fa77f.html | System requirements for AutoCAD LT 2024 |                          41 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
| adsk-8bb54c2d03ff4c3b9798655b090742ea.html | Product Design & Manufacturing Archives |                          41 | trafilatura_short        | very_short_trafilatura_output; very_short_cleaned_content |
