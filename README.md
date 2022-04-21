# SonicNeatAIbeatlevel1
This was an earlier version of my FYP where I used neat algorithm to beat the first level.
Requires Sonic the hedgehog game, which I used from the genesis collection from steam.

-In order to run these AI what you need is to do the following:
Download Visual studio Code
Get the python extention in visual studio code
Open the work space of anyone of the AI below (Their workspaces are in the Demo folder)AI3 which is for Sonic The hedgehog which is can beat the first and second level separately.
Make sure that when you open their workspaces that the interpreter is their respective AI and not the Python of your machine or another environment. You can check this by using CTRL+Shift+P.
Run test_winner in order to see the best agent saved at their time do what they do. For AI3 it contains two winners for the two respective levels.
If you want to train them you can run the badly spelled parallel files for each, however do not near the bottom of each of their files their's a neat.Parrallel(Number,evaluate_genome). In the number area input the amount of threads you want to run the game on. Default I put it on 10 but everyone has a different number of threads on their CPU.

-If for some reason you cannot get visual studio code I recommend doing the following if you're running on a windows system:
Open command line.
CD over to the directory that you place the folder of these files. 
In each of the AI folders there's either a libraries folder or a Test1 folder.
Use the command call Test1/Scripts/activate or if your in a folder that's not AI you call libraries/Scripts/activate.
This will open the venv environment
Now All you have to do is call the agent you want to test or train. Which would be python Demo/test_winner.py

The libraries and technologies that were used throughout this assignment and their licensing.
OpenAI gym Opensourced
OpenAI retro Opensourced
OpenAI Universe Opensourced
Visual Studio Code Microsoft license
Python 3.5-3.7 64bit 
OpenCV-python opensourced
Neat-python Opensourced
NumPy opensourced
SonictheHedgehog Sega
Pickle opensourced
