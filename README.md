****************************************************************************************
***************************  CIS-667-Semester-Project  *********************************
****************************************************************************************
*********************************************************************************************

A small game called "The Game of the Amazons". 


Modified rules: 

              player can skip his turn
              can customize different game board size

Check details about the game rules at https://en.wikipedia.org/wiki/Game_of_the_Amazons

*********************************************************************************************
Requirements and Installation:

python 3.7+：https://www.python.org/downloads/

pygame package: run command "pip install pygame"

pickle: use the "pip install" command

pandas: use the "pip install" command

matplotlib: use the "pip install" command

pytorch: use the "pip install torch" command

numpy: use the "pip install" command


*********************************************************************************************

How to run:

Download the .zip file of the whole project. Use VS code or other IDE. Use VS code as example, 
click open folder, choose the unzipped folder that downloaded from here. 

To play against the minimax and alpha-beta pruning bot, run the file RaceMode.py. A play window
will show up. During the game, player can choose to skip his turn by press "S" on the key board.
There are some outputs in the terminal to show the board status and racing related information.

Run the file TrainMode.py to start training the neural network. After run it, enter the customized
game board size (I recommend board size 5. I tried several different size and found 5 is somehow a 
very stable one) Then it will start the training process. There will be some outputs in the terminal 
show the gameboard and training process. At the end of each game, it will pop out a window with the 
game boards of this round of game. Close that window to continue the training. 

