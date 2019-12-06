# DoggDex
A Pokedex for dogs. 
<br/>

<p align="center">
  <img src="https://images-na.ssl-images-amazon.com/images/I/81twgUcmR1L._AC_SY450_.jpg" alt="butterfree" />
  <br />
</p>


Pokedex is a eletronic devise from de world with the purpose of identify and categorize pokemons from a pictures. Its name comes from mixing the words Pokemon(Poke) and index(dex) and in japonese (the game's original linguage) its name means Pokemon Encyclopedia. So a Pokedex could be defined as a divice that takes photos of Pokemons and categorizes them. As shown in the name our goal in this project is to create a similar software but aimed at dogs(the real life Pokemons). 
<br />


<p align="center">
  <img src="https://res.cloudinary.com/practicaldev/image/fetch/s--_3gm5ojJ--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://cdn-images-1.medium.com/max/2000/1%2A8z04ARQ6AeW6vz0OmRZFUw.png" />
  <br />
</p>


# Table of Contents
- [Solution](#solution)
- [Getting started](#getting-started)
	- [Installation](#installation)
	- [Usage](#usage)
	  - [Generating Melodies](#generating-melodies)
	  - [Adding new genres](#adding-new-genres)
- [Creators](#creators)
- [License](#license)


# Solution

To solve this problem we decided to use a Convolutional Neural Network (CNN), this CNN was implemented using [TensorFlow](https://www.tensorflow.org/), and the chosen language was Python.

## Convolutional Neural Network
CNNs are artificial neural network that is most communly used analize images and its good in finding patterns.

A CNN is has the  basic structure of a generic neural network(input layers, hiden layers and output layers), but CNN hidden layers have a special layer a convolutional layer.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg" alt="butterfree" />
  <br />
  <sub>Basic Neural network structure</sub>
</p>


A convolutinal layer is a layer that has as input an image as a matrix of pixels and has filter, that can be thought as a matrix of size m x n
that is filled with randon numbers, and iterates through the input finding the dot product from the filter and a block of pixels with the same size from the input matrix.




# Getting Started
## Installation
before using the Doggdex you must have intalled in your machine:

 * python3
 * pip3
 
and you must also install or dependences using the command:

```console
pip install -r requirements.txt
```

## Usage
To use the Doggdex you must first train the CNN using the command:

```console
python start.py
```

and then classify the image in the folder /test/test by running the command:


```console
python predict.py
```

# Creators
* [André Filho](https://github.com/andre-filho)
* [Arthur Assis](https://github.com/arthur0496)
* [Guilherme Augusto](https://github.com/guiaugusto)
* [Vitor Falcão](https://github.com/vitorfhc)

# License
DoggDex is licensed under [GPLV3](https://github.com/deeplearningunb/doggdex/blob/master/LICENSE).
