# Tetris Game

<!-- **You must run `chmod a+x tetris_model.py` in order for the program to restart itself!!!** -->

# TODO:

* Fix features in TetrisEngine (line 154: self.last_piece_drop_coords)

# DONE:

* Make it create pieces in batches of 7

# PAPERS

* Temporal Differences-Based Policy Iteration and Applications in Neuro-Dynamic Programming

* Approximate Dynamic Programming Finally Performs Well in the Game of Tetris

* Approximate Modified Policy Iteration

* How to Lose at Tetris

* Approximate Modified Policy Iteration and its Application to the Game of Tetris

* Approximate Policy Iteration Schemes: A Comparison

* Performance Bounds for λ Policy Iteration and Application to the Game of Tetris

* Learning Tetris Using the Noisy Cross-Entropy Method

* Can you win at TETRIS?

* (probably useless) An Algorithmic Survey of Parametric Value Function Approximation

* (probably useless) On the evolution of artificial Tetris players

* (pattern diversity) Distinguishing experts from novices by the Mind’s Hand and Mind’s
Eye

* (features) Improvements on Learning Tetris with Cross Entropy

* (features) A comparison of feature functions for Tetris strategies

* (features) Building Controllers for Tetris

* (features) How to design good Tetris players

* (features) Tetris Agent Optimization Using Harmony Search Algorithm

# USEFUL LINKS:

* tetris drawing tool (for the paper) http://harddrop.com/fumentool

* tutorial/explanation of a similar code repo: http://zetcode.com/gui/pyqt5/tetris/

=============================================================================================

It is a python implementatino of Tetris Game, and a simple AI to play game automatically.

Need python3, PyQt5 and NumPy to be installed.

* `tetris_game.py` is the main application.
* `tetris_model.py` is the data model for this game.
* `tetris_ai.py` is the AI part.

Run `tetris_game.py` from command line and you start to play or watch the AI playing.

```shell
$ python3 tetris_game.py
```

### Play manually

If you want play by yourself, you should uncomment this line in `tetris_game.py`:

```python
# TETRIS_AI = None
```

Or just comment this line:

```python
from tetris_ai import TETRIS_AI
```

Current config could be too fast for human player. So you may want make it slower, by changing value of `Tetris.speed` defined here:

```python
class Tetris(QMainWindow):
    ...
    def initUI(self):
        ...
        self.speed = 10
```

### Play rules

Just like classical Tetris Game. You use *up* key to rotate a shape, *left* key to move left and *right* key to move right. Also you can use *space* key to drop down current shape immediately. If you want a pause, just press *P* key. The right panel shows the next shape.

~ HAVE FUN ~

![Screenshot](doc/pics/screenshot_01.png)
