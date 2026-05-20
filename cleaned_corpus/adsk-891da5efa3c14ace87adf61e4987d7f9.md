---
source_file: "raw_corpus/adsk-891da5efa3c14ace87adf61e4987d7f9.html"
relative_source_path: "adsk-891da5efa3c14ace87adf61e4987d7f9.html"
title: "Spreadsheet to PLC I/O Utility Dialog Box"
cleaned_format: "markdown"
extraction_method: "trafilatura"
raw_char_count: 18268
cleaned_char_count: 4800
---

# Spreadsheet to PLC I/O Utility Dialog Box

Generates a set of PLC I/O drawings from a data file.

Find
Command entry:
**AESS2PLC**

Choose the data file and click Open.

## List of Options

The following options are displayed.

#### Settings

Choose a
*.wdi* file to apply previously saved PLC settings. If you enter the name of the
*.wdi* file, these locations are searched in the following order:

- User folder:
C:\Users\{username}\AppData\Roaming\Autodesk\AutoCAD Electrical {version}\{release}\{country code}\Support\User\
- Active project's .wdp file folder
- Symbol library paths defined for the active project
-
AutoCAD Electrical toolset lookup folder:
C:\Users\{username}\Documents\Acade {version}\AeData\
-
AutoCAD Electrical toolset support (C:\Program Files [(x86)]\Autodesk\AutoCAD {version}\Acade\Support\{language code}\.)
- All paths defined under AutoCAD Options Files Support Files Search Path

Note: If a
*.wdi* file is not used, the settings defined in the
*wdio.lsp* file are used.

- Setup
-
Click Setup to display the Spreadsheet to PLC I/O Utility Setup dialog box where you can define:

- Number and type of ladders
- Number and spacing of rungs
- Reference number style
- Style, scale, and placement of modules
- Placement and spacing for inline devices
- Drawing template for new drawings

Note: If you choose a
*.wdi* file after you make changes in the Spreadsheet to PLC I/O Utility Setup dialog box, the settings in the
*.wdi* file override any changes made.

#### Ladder Reference Numbering

Controls options that relate to line reference numbers on ladders.

- Start
-
Specifies the first line reference number for the first ladder of the first drawing. Leading zeros and embedded alphabetic characters are supported for line reference numbering.

- Index
-
Specifies whether line reference numbers should increment by 1 or by another value.

- Column to column
- Specifies how to calculate the first line reference number of the next ladder column on the same drawing.
Next sequential number: Increment by 1 from the last number of the previous ladder column.

Column to column count: Increment by a specified amount from the first line reference number of the previous ladder column.

- Drawing to drawing
- Specifies how to calculate the first line reference number of the first ladder column on the next drawing.
Next sequential number: Increment by 1 from the last number of the last ladder column on the previous drawing.

Drawing to drawing count: Increment by a specified amount from the first line reference number of the first ladder column on the previous drawing.

#### Module Placement

- Always start at top of ladder
- Starts each I/O module at the top of a ladder, one module for each ladder.
- Same ladder only if module fits
- Builds the module in the ladder with the previous module if it fits completely. If the entire module does not fit, it builds in the next ladder or next drawing.
- Fill ladder - allow module splits
- Builds the module in the same ladder with the previous module and splits the module if necessary. If the module splits, it continues in either the next ladder or next drawing.
- Rungs between
- Skips the specified number of rungs between modules when multiple modules are placed within the same ladder.
- Include unused/extra connections
- Includes the terminals defined as
**When Including Unused** in the PLC database definition for the PLC module.
- Allow pre-defined breaks
- Automatically breaks the PLC module based on any breaks in the PLC database definition for the PLC module.

Drawing File Creation

- Use active drawing
- Indicates to use the active drawing to begin the PLC placement process. If a starting file name has already been entered, this option is not available.
- Starting file name
- Specifies the drawing file to begin the PLC placement process. The .dwg extension is not required. The drawing is saved in the same folder as the .wdp file for the active project.
- Pause between drawings
- Displays a dialog box before each new drawing is generated, which enables you to adjust the settings.
- Free run
- Generates all PLC drawings without stopping.
- Sheet
- Enter a value for the drawing sheet number. Sheet numbers increment for each new drawing.
- Add new drawing to active project
- Adds newly created drawings to the active project. The new drawings are added to the end of the project’s drawing list.

Save

Saves the setup information and settings in a
*.wdi* file to reuse. Settings include the options in this dialog box, the Spreadsheet to PLC I/O Utility Setup dialog box, and the Spreadsheet to PLC I/O Drawing Generator dialog box.

Note: Default settings can be defined in the source file,
*wdio.lsp*. Open the file with a text editor and edit the values near the top of the file.
