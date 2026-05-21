---
source_file: "raw_corpus/adsk-0f7179f7c5eb451da74dd9cdb79bee5a.html"
relative_source_path: "adsk-0f7179f7c5eb451da74dd9cdb79bee5a.html"
title: "Stop & Restart Autodesk License Server | Autodesk"
cleaned_format: "markdown"
extraction_method: "trafilatura"
document_language: "en"
document_language_name: "English"
document_language_confidence: 1.0
heading_count: 7
subheading_count: 6
headings: "h1: Stop & Restart Autodesk License Server | Autodesk | h2: From a Windows license server | h3: Stop the server | h3: Restart the server | h2: From a macOS or Linux license server | h3: Stop the server | h3: Restart the server"
subheadings: "From a Windows license server | Stop the server | Restart the server | From a macOS or Linux license server | Stop the server | Restart the server"
raw_char_count: 130740
cleaned_char_count: 1099
tfidf_keyword_count: 12
tfidf_keywords: "server | stop | license server | license | restart | lmtools | start | tab | enter following | terminal | license file | force"
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
