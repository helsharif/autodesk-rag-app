---
source_file: "raw_corpus/adsk-e0ea873f6e554cb4bdba2b846cac0ee8.html"
relative_source_path: "adsk-e0ea873f6e554cb4bdba2b846cac0ee8.html"
title: "About Display System"
cleaned_format: "markdown"
extraction_method: "trafilatura"
raw_char_count: 8068
cleaned_char_count: 2401
---

# About Display System

The display system in AutoCAD Architecture 2022 toolset is designed so that you only have to draw an architectural object once. The appearance of that object then changes automatically to meet the display requirements of different types of drawings, view directions, or levels of detail.

The view-dependent display of objects in AutoCAD Architecture 2022 toolset is made possible by a hierarchical system of display settings that specify display properties (visibility, layer, color, linetype, and so on) for individual display components of all the different types of architectural objects under all the different viewing scenarios.

For example, a door object has 5 display components by default: Door Panel, Frame, Stop, Swing, and Glass. (Many display components correspond directly to the physical components of the object, but some, like Swing, do not.) For each of these display components, the drawing default settings specify the relevant display properties as appropriate for the drawing type. In the case of doors, the default settings specify that the Swing component is visible in plan and elevation views, but not in 3D views.

The drawing default settings for a particular type of object apply to all such objects in the drawing, except those for which an override is in effect. For example, you can change a setting for all doors of a particular style (a style override) or for an individual door (an object override).

If you are a CAD manager, you will want to fully understand the display system structure and the display manager so that you can modify and organize default settings as necessary to implement your own display standards. But any user can quickly change the appearance of an object in a particular view by modifying values on the Display tab of the Properties palette.

Display tab when display component is selected

To change the display using this tab, click (Select Components), select an object display component (like a hatch or a boundary), and then select or enter a new value for the display property you want to change (such as color, visibility, or lineweight).

The results are immediately visible in the drawing area for the current display representation and can be applied to other display representations that use the same component. You can also apply style or object overrides by changing the value of Display controlled by.
