# pytorch-lightning-fine-tuning

Example of using Pytorch Lightning to fine tune a pretrained Causal Language Model. This is designed to run quickly on a single GPU machine. It uses the relatively small [CMU Book Summary Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html) and is  set to run with a single training epoch by default, to make it quicker to run.

## Increasing the number of epochs

Change the `max_epochs` parameter in the `Trainer` call.

```
trainer = Trainer(max_epochs=10, devices=1, accelerator="gpu")
```


## Using on a multiple GPU machine

To run this on multiple GPUs, you can alter the `devices` parameter in the Pytorch Lightning `Trainer` call:

```
trainer = Trainer(max_epochs=1, devices=2, accelerator="gpu")
```
