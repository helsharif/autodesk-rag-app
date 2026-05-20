---
source_file: "raw_corpus/adsk-0f7179f7c5eb451da74dd9cdb79bee5a.html"
relative_source_path: "adsk-0f7179f7c5eb451da74dd9cdb79bee5a.html"
title: "Stop & Restart Autodesk License Server | Autodesk"
cleaned_format: "markdown"
extraction_method: "trafilatura"
raw_char_count: 130740
cleaned_char_count: 1099
---

# Stop & Restart Autodesk License Server | Autodesk

To perform system maintenance on a license server, including uninstalling Network License Manager, first stop the license server. When you finish maintenance work, restart the license server.

## From a Windows license server

### Stop the server

- From the Start menu, search for LMTOOLS.
- In LMTOOLS, click the Service/License File tab.
- Select Configure Using Services.
- Select the service name for the license server you want to stop manually.
- Click the Start/Stop/Reread tab.
- Stop the server. The best practice is to select Force Server Shutdown before clicking Stop Server. Then wait five seconds before attempting to start the server again.

### Restart the server

On the Start/Stop/Reread tab in LMTOOLS, click Start Server.

## From a macOS or Linux license server

### Stop the server

Enter the following in Terminal:

./lmutil lmdown -q -force

### Restart the server

Enter the following in Terminal, replacing *acad.lic* with your license file name and *debug.log* with your log file name:

./lmgrd -c acad.lic -l debug.log
