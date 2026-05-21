---
source_file: "raw_corpus/adsk-f0ef4d59f8b047dc89a88db1feb9e8cb.html"
relative_source_path: "adsk-f0ef4d59f8b047dc89a88db1feb9e8cb.html"
title: "Deploy content to collections | Autodesk"
cleaned_format: "markdown"
extraction_method: "trafilatura"
document_language: "en"
document_language_name: "English"
document_language_confidence: 1.0
heading_count: 3
subheading_count: 2
headings: "h1: Deploy content to collections | Autodesk | h2: Distribute a deployment based on the application model | h2: Distribute a deployment based on the package model"
subheadings: "Distribute a deployment based on the application model | Distribute a deployment based on the package model"
raw_char_count: 134817
cleaned_char_count: 5452
tfidf_keyword_count: 12
tfidf_keywords: "device collection | select | client | deployment | device | package | option | cache | deploy | application | collection | clients"
---

# Deploy content to collections | Autodesk

## Distribute a deployment based on the application model

With the Application model, you distribute the deployment to device collections and user collections.

- Select the application you want to deploy from:
*\Software Library\Overview\Application Management\Applications\* - Click Deploy in the navigation bar to start the Configuration Manager distribution process.
- Start the collection you want to use:
- User collection signed in to client machines
- Online devices in the selected device collection

- Select a point where to distribute the application, unless you've already distributed it.
- Choose whether you want to make the application available for users to install or to automatically push and install it on the clients.
- Available. You can require that users obtain administrator approval before installing the product. A notification will appear in the Software Center, where the administrator must approve the installation.
- Required. If you also select Pre-deploy software to the user's primary device, the application is installed on the client designated as the user's primary device.

- (Optional) Select when the application will be available.
- Specify which messages the Configuration Manager displays on the client.
- Determine whether to notify the System Center Operations manager when a product is installed or fails.
- Review the Summary page before committing the deployment.
- Click Next to deploy.

## Distribute a deployment based on the package model

Deployments based on the Package model distribute content to devices, not users.

- After you select your package in the Software Center and start the deployment wizard, select Device Collection from the drop-down menu.
- Select the device collection to deploy. Make sure that you deploy to the device collection you created for the intended client machines.
- Choose a deployment purpose setting:
- Available. Select this option if you want to make the product available for the user to install from the Configuration Manager. This option makes the package available to the clients in the device collection. But the user must choose to install it.
- Required. Select this option if you want to push and install the product on all devices in the device collection.

- Consider options for waking up clients and continuing installation after the deadline:
- Send wake-up packets. Select this option to start up clients (if they are turned off) and install the package on them. (Client machines must be configured for this option.)
- Allow clients on a metered internet connection to download content after the installation deadline. Allow clients on slow networks to continue the installation even if the deadline elapses. (This option may incur extra costs.)

- Choose Scheduling options. These options vary according to whether you selected Available or Required in the previous step.
- If you selected Available: Specify the time and date you make the package available for users to install on their clients. Also indicate whether and when package availability expires. These options are not required.

- If you selected Required: Create an assignment schedule. The quickest option is to select New and then choose to deploy as soon as possible.

- Select a Rerun option, which determines when the installer will rerun.
- Never rerun deployed program. The program will never rerun on a specific client under any circumstances.
- Always rerun program. Tells the client to disregard previous execution status, such as removing a client and adding it back to a collection.
- Rerun if failed previous attempt. Rerun determined by return codes.
- Rerun if succeeded on previous attempt. The program runs again only if it has previously run successfully on the client. This setting is useful for recurring updates, which require the installation of a previous update before the installation of a later one.

- Choose a Distribution Points deployment option:
- Download content from distribution point and run locally. If you select this option, content files are downloaded to C:\Windows\ccmcache\ by default, but you can change the location.
Run the program from the distribution point. If you select this option, the installer runs remotely from the distribution point instead of downloading and running from the local cache. You don't have to set the cache size, but you do need to verify that you have enough free storage space.

Note: To download the content and run it locally, make sure that the client cache has enough space. (To verify the cache size, go to Control panel > Configuration Manager > Cache.) To check for logged errors about cache limits, go to

*CAS.log*in the*C:\Windows\CCM\Logs\*folder. Also, there must be enough free storage space for the content to be downloaded and extracted locally. - Review the Summary page to verify your deployment selections. You can go back to edit your settings.
- Finish the deployment and view the Completion page. Your package is now deployed to the selected device collection.

Warning: This "run locally" setting is not recommended and doesn't work for Application and Hybrid mode deployments.

Next topic: [Distribute product updates](https://aem-efddotcom-author.efddotcom.autodesk.com/content/autodesk/global/en/support/download-install/admins/create-deployments/installing-deployments-with-microsoft-configuration-manager/distribute-product-updates.html)
