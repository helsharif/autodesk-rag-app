---
source_file: "raw_corpus/adsk-aea0ca0a7bbe4f84b1a33bceb54c7673.html"
relative_source_path: "adsk-aea0ca0a7bbe4f84b1a33bceb54c7673.html"
title: "Autodesk Options File | Administrator Installation"
cleaned_format: "markdown"
extraction_method: "trafilatura"
raw_char_count: 147642
cleaned_char_count: 15452
---

# Autodesk Options File | Administrator Installation

Access to licenses may be controlled with an options file configured with the settings you want to use. Use of the options file is optional and isn't required for normal license operation. For complete information about options files, see Managing the Options File in the FLEXnet Publisher License Administration Guide PDF file installed with NLM. To view that PDF guide, see the Network License Manager folder on your license server.

The following sections describe how to create an options file and perform various functions with it. A sample options file is displayed in the final section.

## Create an options file

You can create option files for one or more servers and manage network licenses through those files.If you don’t need the capabilities of the options file, you don't have to create options files for all servers.

If your options file defines controls based on product features, the controls apply to products whether you are on a maintenance plan or multi-user subscription.However, if you define controls at the product package level, you can separate maintenance plan seats from multi-user subscriptions.

To activate an options file, save it and then reread the license file.

- Open a text editor, such as Notepad (Windows) or TextEdit (macOS). Enter the commands and parameters as specified for the report log or other operations.
- Name the file
*adskflex.opt*and save it where you saved the network license file.**Note:**Be sure to save the file with the extension .opt. NLM does not recognize .txt. - Reread the license file on the server.
- For a Windows server, use the LMTOOLS utility installed with NLM and select Reread License File.
- For a macOS or Linux server, open a Terminal window, navigate to the Autodesk Network License Manager folder (flexnetserver), and enter the following string:
./lmutil lmreread -c @hostname-all

- Check the log file to be sure that the options file has been read. If it wasn’t, stop and restart the license server.

## Set a license timeout

Set a limit on how long a license is allocated to an inactive computer before the server reclaims it. If the computer becomes active again, the server issues a fresh license. If a license isn’t available, the user gets an alert.

Open the options file in a text editor and enter one of the following commands on a separate line.

TIMEOUT feature_code n

Here, **feature_code** is the name of the product and **n** is the number of seconds of inactivity before the server reclaims a license.

TIMEOUT 66800REVIT_F 7200

This example indicates that the license timeout for Revit is 7200 seconds, or two hours.

TIMEOUTALL n

Here, the timeout is for all products, where **n** is the number of seconds before the server reclaims an inactive Autodesk product.

## Configure license borrowing

There are several options to control license borrowing:

- Maintain a pool of available licenses
- Set the borrowing period
- Exclude license borrowing
- Include license borrowing

### Maintain a pool of available licenses

BORROW_LOWWATER specifies how many licenses cannot be borrowed. This option ensures that the specified number of licenses always remains in the license pool. In this way, the pool isn't completely depleted by borrowing.

BORROW_LOWWATER [productfeature] [count]

The following example blocks license borrowing for five AutoCAD subscription licenses with multi-user access:

BORROW_LOWWATER 64300ACD_T_F 5

### Set the borrowing period

You can specify the maximum time interval, in hours, that a specific license can be borrowed. This value can't exceed the maximum borrow period specified by Autodesk in the product license file.

MAX_BORROW_HOURS [productfeature] [hours]

The following example limits license borrowing of an AutoCAD subscription license with multi-user access to 3 days:

MAX_BORROW_HOURS 64300ACD_T_F 72

Note: Without this setting in an options file, the maximum borrow period for licenses is 6 months or upon expiration of the subscription with multi-user access, whichever is sooner. We strongly recommend establishing a shorter borrow period.

### Exclude license borrowing

Block the ability to borrow specific licenses. Any users, hosts, or IP addresses not explicitly excluded can continue to borrow licenses.

EXCLUDE_BORROW [productfeature] [type] [name]

The following examples block borrowing an AutoCAD subscription license with multi-user access for a user, computer, group, and so on:

EXCLUDE_BORROW 64300ACD_T_F USER smithj EXCLUDE_BORROW 64300ACD_T_F HOST computer1 EXCLUDE_BORROW 64300ACD_T_F GROUP EngineeringGroup EXCLUDE_BORROW 64300ACD_T_F HOST_GROUP DraftingDept EXCLUDE_BORROW 64300ACD_T_F INTERNET 192.168.0.100 EXCLUDE_BORROW 64300ACD_T_F INTERNET 192.168.0.* EXCLUDE_BORROW 64300ACD_T_F PROJECT CivilProject

Note: As with EXCLUDE and INCLUDE, EXCLUDE_BORROW takes precedence over conflicting INCLUDE_BORROW statements.

### Include license borrowing

Allow borrowing of specific licenses. All users, hosts, or IP addresses not explicitly included are blocked from borrowing the specified licenses. If you want to block license borrowing for only a few users, consider using EXCLUDE_BORROW instead.

INCLUDE_BORROW [productfeature] [type] [name]

The following examples allow borrowing an AutoCAD subscription license with multi-user access for a user, computer, group, and so on:

INCLUDE_BORROW 64300ACD_T_F USER smithj INCLUDE_BORROW 64300ACD_T_F HOST computer1 INCLUDE_BORROW 64300ACD_T_F GROUP EngineeringGroup INCLUDE_BORROW 64300ACD_T_F HOST_GROUP DraftingDept INCLUDE_BORROW 64300ACD_T_F INTERNET 192.168.0.100 INCLUDE_BORROW 64300ACD_T_F INTERNET 192.168.0.* INCLUDE_BORROW 64300ACD_T_F PROJECT CivilProject

Note: Because an EXCLUDE_BORROW statement always takes precedence over a conflicting INCLUDE_BORROW statement, best practice is to use only one of these statements in a single options file.

## Use package and feature codes

The following examples show settings for options file parameters for a specific product using that product's package code. For example, the following statement uses the package code from a subscription with multi-user access license for AutoCAD 2016 (64300ACD_T_F) to reserve one license of AutoCAD 2016-2013 for a specific user:

RESERVE 1 64300ACD_T_F USER smithj

By using the package code, you apply that options file parameter to all eligible versions of the product according to the previous version rights of the subscriber.

Sometimes you may use package codes in your options file and your license file may contain both perpetual or maintenance plan and subscription licenses with multi-user access for the same product. In this case, you need to include other parameters to accommodate the subscription licenses with multi-user access. The following statements reserve one license of AutoCAD 2016-2013 for a specific user, whether it is a perpetual license on a maintenance plan or a subscription license with multi-user access:

RESERVE 1 64300ACD_F USER smithj RESERVE 1 64300ACD_T_F USER smithj

After you add parameters for subscription licenses with multi-user access, you don't need to modify the options file when the subscription is renewed or a version is released.

Use a product feature code (for example, 86445ACD_2016_0F) only if you are setting an options file parameter for a perpetual license that is not on a maintenance plan. For perpetual licenses on a maintenance plan and subscription licenses with multi-user access, always use the package code. The following statement uses a feature code to reserve five seats of a perpetual license of AutoCAD 2016 (not on a maintenance plan) for a specific group:

RESERVE 5 86445ACD_2016_0F GROUP EngineeringGroup

## Enter comments in the options file

The license manager ignores all syntax after a hash (#) symbol.

#This is a comment

## Define groups

As you specify who can and can't access licenses, it’s convenient to define groups of users or computers. Groups are useful when you reserve or restrict license usage.

You can define groups using the Windows sign-in name or the computer name. By default, computer names and usernames are case-sensitive unless you added the GROUPCASEINSENSITIVE ON statement to the options file.

GROUP [groupname] [user1] [user2] [user3]

HOST_GROUP [groupname] [computername1] [computername2] [computername3]

The following example defines a group called EngineeringGroup for three users:

GROUP EngineeringGroup smithj jonesb whitef

The following example defines a group called DraftingDept with three computers:

HOST_GROUP DraftingDept computer1 computer2 computer3

To create a group with many users, define multiple GROUP lines with the same group name, each containing up to a maximum of 4,000 characters. If you define multiple GROUP lines with the same group name, you can add all the specified users to a single group.

## Disable case sensitivity for user and computer names

You can enable/disable case sensitivity for usernames and computer names when you use the GROUP or HOST_GROUP commands. By default, user and computer names are case sensitive.

The GROUPCASEINSENSITIVE statement disables (ON) and enables (OFF) case sensitivity for user and computer names:

GROUPCASEINSENSITIVE ON

GROUPCASEINSENSITIVE OFF

## Reserve licenses

Reserve a specific number of licenses to ensure that product licenses are available when needed. (Reserved licenses aren't available to other users.) For example, you can reserve licenses for people working on a time-sensitive project.

RESERVE [count] [productfeature] [type] [name]

The following examples reserve either one or five seats for an AutoCAD subscription license with multi-user access for a user, a computer, a group, and so on:

RESERVE 1 64300ACD_T_F USER smithj RESERVE 1 64300ACD_T_F HOST computer1 RESERVE 5 64300ACD_T_F GROUP EngineeringGroup RESERVE 5 64300ACD_T_F HOST_GROUP DraftingDept RESERVE 1 64300ACD_T_F INTERNET 192.168.0.100 RESERVE 5 64300ACD_T_F INTERNET 192.168.0.* RESERVE 5 64300ACD_T_F PROJECT CivilProject

## Restrict maximum license use

Limit the use of licenses to maximize license availability by restricting access to a fixed number of licenses for specified products.

MAX [count] [productfeature] [type] [name]

The following examples set a limit of either one or five seats for an AutoCAD subscription license with multi-user access for a user, computer, group, and so on.

MAX 1 64300ACD_T_F USER smithj MAX 1 64300ACD_T_F HOST computer1 MAX 5 64300ACD_T_F GROUP EngineeringGroup MAX 5 64300ACD_T_F HOST_GROUP DraftingDept MAX 1 64300ACD_T_F INTERNET 192.168.0.100 MAX 5 64300ACD_T_F INTERNET 192.168.0.* MAX 5 64300ACD_T_F PROJECT CivilProject

## Exclude license usage

Block access to specific licenses. All users, hosts, or IP addresses that aren't explicitly excluded have access to these licenses.

EXCLUDE [productfeature] [type] [name]

The following examples block access to an AutoCAD subscription license with multi-user access for a user, computer, group, and so on:

EXCLUDE 64300ACD_T_F USER smithj EXCLUDE 64300ACD_T_F HOST computer1 EXCLUDE 64300ACD_T_F GROUP EngineeringGroup EXCLUDE 64300ACD_T_F HOST_GROUP DraftingDept EXCLUDE 64300ACD_T_F INTERNET 192.168.0.100 EXCLUDE 64300ACD_T_F INTERNET 192.168.0.* EXCLUDE 64300ACD_T_F PROJECT CivilProject

Note: EXCLUDE statements always supersede conflicting INCLUDE statements. If there is a conflict, the EXCLUDE statement takes precedence.

## Include license usage

Give access to specific licenses. This setting blocks all users, hosts, or IP addresses that aren't explicitly included. If you want to block only a few users, consider using EXCLUDE instead.

INCLUDE [productfeature] [type] [name]

The following examples give access to an AutoCAD subscription license with multi-user access for the specified a user, computer, group, and so on:

INCLUDE 64300ACD_T_F USER smithj INCLUDE 64300ACD_T_F HOST computer1 INCLUDE 64300ACD_T_F GROUP EngineeringGroup INCLUDE 64300ACD_T_F HOST_GROUP DraftingDept INCLUDE 64300ACD_T_F INTERNET 192.168.0.100 INCLUDE 64300ACD_T_F INTERNET 192.168.0.* INCLUDE 64300ACD_T_F PROJECT Civil Project

Note: Because an EXCLUDE statement always takes precedence over a conflicting INCLUDE statement, best practice is to use only one of these statements in a single options file.

## Create a report log

The report log file is a compressed, encrypted file that generates usage reports on license activity.

REPORTLOG [+]report_log_path

Windows example: A report log named report.rl is in the folder C:\My Documents.

REPORTLOG +"C:\My Documents\report.rl"

macOS or Linux example: A report log named report.rl is in the folder /Users//NLM.

REPORTLOG +"/Users/<user id>/NLM/report.rl"

Note: Paths that contain spaces must be in quotation marks. The path must already exist. The license manager can't create this location for you.

In the REPORTLOG syntax, [+] means that new entries to the log file are appended to previous entries rather than overwriting them each time the Network License Manager restarts. A best practice is to use the [+] option to retain a history of log entries.

## Define product association

You can define specific computers as part of a project by configuring an environment variable on those client machines. Then use this project designation& to control license access through the options file. The project environment variable is optional. You don't have to specify it when you define groups directly in the options file.

To designate a computer as a member of a project:

- On the Start menu (Windows), click Settings > Control Panel.
- In Control Panel, double-click System.
- In the System properties dialog box, click the Advanced tab.
- On the Advanced tab, click Environment Variables.
- Under System Variables, click New.
- In the New System Variable dialog box, enter LM_PROJECT for the Variable name and the project name for the Variable value.
- Click OK to close each dialog box.

Your project is now defined. You don't have to restart the computer for these settings to take effect.

## Options file example

# Company A - Options File # # Last updated: Jan 5, 2021 by John Smith # Sets inactivity timeout for all products to 90 minutes: TIMEOUTALL 5400 # Sets maximum borrow time of 10 days for a subscription with multi-user access of AutoCAD: MAX_BORROW_HOURS 64300ACD_T_F 240 # Defines a report log: REPORTLOG +”C:\adsk_flexnet\logs\adskflex_report.log”` # Disables case sensitivity when defining groups/host_groups: GROUPCASEINSENSITIVE ON # Defines a Civil Eng group and controls license usage: GROUP CivilTeam smithj jonesb whitef MAX 3 64300ACD_T_F GROUP CivilTeam RESERVE 3 64900CIV3D_T_F GROUP CivilTeam INCLUDE_BORROW 64900CIV3D_T_F HOST_GROUP CivilTeam # Defines a Mech Eng group by computer name and controls license usage: HOST_GROUP MechTeam JoesPC FredsPC WillsPC JohnsPC BobsPC MAX 5 64300ACD_T_F HOST_GROUP MechTeam RESERVE 5 85578INVNTOR_T_F HOST_GROUP MechTeam RESERVE 2 65500ACAD_E_T_F HOST_GROUP MechTeam INCLUDE_BORROW 85578INVNTOR_T_F HOST_GROUP MechTeam INCLUDE_BORROW 65500ACAD_E_T_F HOST_GROUP MechTeam # Controls license usage for general drafters in subnet 192.168.0.* RESERVE 30 64300ACD_T_F INTERNET 192.168.0.* EXCLUDE 64900CIV3D_T_F INTERNET 192.168.0.* EXCLUDE 85578INVNTOR_T_F INTERNET 192.168.0.* EXCLUDE 65500ACAD_E_T_F INTERNET 192.168.0.* EXCLUDE_BORROW 64300ACD_T_F INTERNET 192.168.0.*
