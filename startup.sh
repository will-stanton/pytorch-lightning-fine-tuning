#!/bin/bash
pip install datasets transformers[torch] accelerate lightning
wget https://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz
tar -xvzf booksummaries.tar.gz