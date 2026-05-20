# Sample Cleaning Quality Review

Random seed: `20260520`  
Sample size: `100` successful cleaned documents

## High-level result

- Documents with no quality flags: `25`
- Documents with one or more flags: `75`
- Median cleaned/raw visible-text ratio: `0.548`
- Median cleaned character count: `1,700`
- Sample raw tables: `6`
- Sample Markdown tables retained: `6`

## Extraction methods

- `trafilatura`: `47`
- `trafilatura_short`: `28`
- `beautifulsoup_rich_fallback`: `14`
- `beautifulsoup_fallback`: `8`
- `beautifulsoup_table_fallback`: `3`

## Quality flags

- `heading_loss`: `42`
- `very_short_output`: `34`
- `low_visible_text_retention`: `8`
- `punctuation_noise`: `4`
- `possible_boilerplate_remaining`: `2`

## Worst flagged sample rows

| relative_path                              | title                                                              | extraction_method_used   |   raw_visible_chars |   cleaned_chars |   raw_tables |   md_tables |   raw_headings |   md_headings |   cleaned_to_raw_visible_ratio | quality_flags                                             |
|:-------------------------------------------|:-------------------------------------------------------------------|:-------------------------|--------------------:|----------------:|-------------:|------------:|---------------:|--------------:|-------------------------------:|:----------------------------------------------------------|
| adsk-a124ea70637f478c8adeecbd5e1a61f9.html | Business Archives | Autodesk News                                  | trafilatura_short        |                4196 |              19 |            0 |           0 |             10 |             1 |                         0.0045 | very_short_output;low_visible_text_retention;heading_loss |
| adsk-0966028aeb674e83add24a3ba3a4361d.html | Sustainability in AEC Archives | Autodesk News                     | trafilatura_short        |                4204 |              32 |            0 |           0 |             10 |             1 |                         0.0076 | very_short_output;low_visible_text_retention;heading_loss |
| adsk-f0702460015846dda0f1065e8e2be249.html | Inventor Engineer-to-Order | Autodesk                              | trafilatura_short        |                3899 |              39 |            0 |           0 |             10 |             1 |                         0.01   | very_short_output;low_visible_text_retention;heading_loss |
| adsk-bec4d214546348879b18d76017a6f750.html | Autodesk Advanced Manufacturing Resources Library & Content Center | trafilatura_short        |                3959 |             101 |            0 |           0 |             12 |             1 |                         0.0255 | very_short_output;low_visible_text_retention;heading_loss |
| adsk-229bb184cfd5457e944171ba7f597960.html | Site Selector                                                      | trafilatura_short        |                1336 |              15 |            0 |           0 |              5 |             1 |                         0.0112 | very_short_output;heading_loss                            |
| adsk-943b27df40db492793f89e9634def031.html | Site Selector                                                      | trafilatura_short        |                1336 |              15 |            0 |           0 |              5 |             1 |                         0.0112 | very_short_output;heading_loss                            |
| adsk-a50257da69f948c3847bf1763a250fd3.html | Forma Webinar                                                      | beautifulsoup_fallback   |                 328 |              93 |            0 |           0 |              3 |             3 |                         0.2835 | very_short_output;punctuation_noise                       |
| adsk-036657b896d64873aeb89c0b32e33844.html | Install Autodesk Products | Administrator Installation             | beautifulsoup_fallback   |                1937 |              98 |            0 |           0 |              9 |             1 |                         0.0506 | very_short_output;heading_loss                            |
| adsk-8294ad72ef69483087a0e0e747a15067.html | Digital Factory | Webinar | Autodesk                               | beautifulsoup_fallback   |                 351 |             116 |            0 |           0 |              3 |             3 |                         0.3305 | very_short_output;punctuation_noise                       |
| adsk-9a9e0aaeecd9477da550a5b361c6ed32.html | Q3 FY24 Autodesk Earnings Conference Call | Autodesk, Inc.         | trafilatura_short        |                1083 |             240 |            0 |           0 |              8 |             1 |                         0.2216 | very_short_output;heading_loss                            |
| adsk-8cf0c412f1c848988e72aaf988fe1489.html | Privacy Statement Russian                                          | trafilatura              |               34805 |            5769 |            0 |           0 |             19 |             1 |                         0.1658 | heading_loss;possible_boilerplate_remaining               |
| adsk-e4c92773fe2c40fc83e28ad83b1ee753.html | Help                                                               | trafilatura_short        |                   4 |               6 |            0 |           0 |              0 |             1 |                         1.5    | very_short_output                                         |
| adsk-e08af7ff3aea46cea0af46e855ec2897.html | Help                                                               | trafilatura_short        |                   4 |               6 |            0 |           0 |              0 |             1 |                         1.5    | very_short_output                                         |
| adsk-cede02128f824713851aa1141f146165.html | Help                                                               | trafilatura_short        |                   4 |               6 |            0 |           0 |              0 |             1 |                         1.5    | very_short_output                                         |
| adsk-2401a58935e84b7e8f681064ea9cf3d1.html | Overview | Revit | Autodesk                                        | trafilatura_short        |                  27 |              10 |            0 |           0 |              0 |             1 |                         0.3704 | very_short_output                                         |

## Interpretation

The table-loss and low-retention flags are the highest-priority indicators. After the cleaner update, the sample shows better retention on content-rich pages while still leaving deliberately thin pages, login/account shells, and JS-only help stubs very short. Remaining punctuation-noise rows should be manually inspected before adding another broad rule.