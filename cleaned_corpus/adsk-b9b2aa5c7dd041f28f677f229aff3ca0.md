---
source_file: "raw_corpus/adsk-b9b2aa5c7dd041f28f677f229aff3ca0.html"
relative_source_path: "adsk-b9b2aa5c7dd041f28f677f229aff3ca0.html"
title: "About Creating and Editing Layer Standards"
cleaned_format: "markdown"
extraction_method: "trafilatura"
document_language: "en"
document_language_name: "English"
document_language_confidence: 1.0
heading_count: 1
subheading_count: 0
headings: "h1: About Creating and Editing Layer Standards"
subheadings: ""
raw_char_count: 7473
cleaned_char_count: 1994
tfidf_keyword_count: 12
tfidf_keywords: "layer | layer standard | standard | layer standards | discipline | value | standards | major | rules | names | field | edition"
---

# About Creating and Editing Layer Standards

A layer standard contains predefined layer names and a set of rules that determines the names of new layers that you create using that layer standard. You can use layer standards to establish individual, project, or office layering conventions that provide consistent and informative layer names. AutoCAD Architecture 2022 toolset provides a number of layer standards that you can use in drawings. You can also create your own layer standards by customizing an existing layer standard.

When you create a new layer using a layer standard, the layer name has a number of parts separated by delimiters (for example, hyphens). Each part of the layer name is determined by rules specified in a corresponding field of the layer standard.

For example, the AIA 2nd Edition layer standard includes five fields that form each new layer name: Discipline Designator, Major, Minor 1, Minor 2, and Status. Each field is separated by a hyphen (?) delimiter, as in the following example:

(Discipline Designator) - (Major) - (Minor 1) - (Minor 2) - (Status)

A layer in your drawing with a Discipline value of “A,” a Major value of “Wall,” a Minor 1 value of “Full,” a Minor 2 value of “Abov” and a Status value of “D,” to denote a demolition layer, would be named A-WALL-FULL-ABOV-D.

You can edit the layer standard definition to change these rules.

You can override the information in each field to change the way a layer standard creates a layer name by specifying layer key overrides.

AutoCAD Architecture 2022 toolset provides industry-standard layering conventions including: AIA 2nd Edition, BS1192 Descriptive, and BS1192 - AUG Version 2. Additional international layer conventions are also provided including: DIN 276, ISYBAU Long Format, ISYBAU Short Format, and STLB. Each layer standard contains specific information organized in fields. You can specify how the information appears in each field by changing the values in the layer standard fields
