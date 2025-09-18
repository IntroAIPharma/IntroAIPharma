# Deep Generative Chemistry for Drug Design
Here we share the notebooks for the workshop on molecule generation used for the 
courses: 

 - *Introduction to AI in Pharmacy - PHA-6935*, offered in Fall 2023.
   - Updated for PHA 6241, Fall 2025.
 - *Drug Design II - PHA 6467*, offered in Spring 2024

## HiPerGator Instructions

Students must login via OnDemand, but directing the browser to https://ood.rc.ufl.edu/, and logging in. 
Once in the main page, choose to start a Jupyter Notebbok.

A new page will open. In this one, adjust the requirements before starting the process:

- Additional Jupyter Arguments: enter `--notebook-dir=/blue/pha6241/<userid>`, substituting `<userid>` by your GatorLink ID.
  - Note that this shows `pha6241`, the group for the class in Fall 2025, if you have a different group, please use that instead.
- Number of CPU cores requested per MPI task: enter `4`
- Maximum memory requested for this job in Gigabytes: enter `24`
- Time Requested for this job in hours: enter `2` (You can request for more hours if you need.)
- Generic Resource Request enter `gpu:1`

After setting all these, click "launch". In the new window, click "Connect to Jupyter". This will open Jupyter Lab.

## Clone the Git repository
- On the left side, click the Git symbol
- Click "Clone a repository"
- on the pop-up, click "Download the repository", then enter the address: "https://github.com/IntroAIPharma/IntroAIPharma"
- Click OK
It will download the repository, and you should see "IntroAIPharma" on the left side bar.

## Create your kernel from pre-configured environment

- Open a Terminal, either in your Jupyter session, or from the OOD **Clusters menu** > "**>_ HiPerGator Shell Access**".
- Enter the command below to create a kernel:
  
      /data/reference/class/IntroAIPharma/AI_pharma/bin/python -m ipykernel install --user --name IntroAIPharma --display-name IntroAIPharma

- When you run that command, the output should be: `Installed kernelspec IntroAIPharma in /home/GatorLink/.local/share/jupyter/kernels/introaipharma` (with your GatorLink shown)
  

## Open the Notebook

After you clone the repository, you should see the "IntroAIPharma" folder in the navigation pane on the left.
(If you don't see it, make sure the navigation pane is active by clickin in the folder image in the top left).

Navigate to IntroAIPharma --> Workshop then double click to open the notebook.

Have fun.

