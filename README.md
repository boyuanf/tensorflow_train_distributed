# tensorflow_train_distributed

Almost the same version as tensorflow_train_distributed_AWS, but runs on windows.

Master branch read data by single thread mnist.train.next_batch(BATCH_SIZE).

ReadByQueue branch read data by mutiple thread and queue, works fine if you have GPU on the server, otherwise will stole too many CPU power from computing the gradient, then slow down the training process.
