THIS IS A RANDOM SEARCH GPU PROGRAM FOR BLOOMFILE

you can download the list of all funded BTC addresses here http://addresses.loyce.club/

then by using python bloom.py convert it from .txt to .bin - that will be used in this program

1) Create a BloomFile from your addrs.txt

`python bloom.py create addrs.txt addrs.bin --size 300` - the higher the size, the less false positives

2) Check if cuda finds the hashes160 by running

`main.exe test addrs.bin 29a78213caa9eea824acf08022ab9dfc83414f56` - verify puzzle 21 hash is there\
`main.exe test addrs.bin 4e15e5189752d1eaf444dfd6bff399feb0443977` - verify puzzle 33 hash is there\

3) Run the search

`main.exe 100000 1fffff addrs.bin` - search for puzzle 21\
`main.exe 100000000 1ffffffff addrs.bin` - search for puzzle 33\


4) Run The global search for any BTC

`main.exe 1 ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff addrs.bin` - will randomize a private key from 1 to ffffffffff...

in case if this was useful to you, can you please donate BTC:

`bc1q8n38pk3urztlt4vceq0h089l9jdcw58l2c0e80`

so i will be more motivated to develop further this project

author: https://t.me/biernus
