# Linear Regression in C
Given a dataset of tuples $(n, 2n) \in \mathbb{N}$, the regression model $y = xw$ will estimate the parameter $w$.

## Content (sorted by interest)
- [Results](#results)
- [Theorical loss](#theorical-loss)
- [Training](#training)
- [Dataset generation](#dataset-generation)

## Results
```bash
TRAIN_SIZE = 10
THEORICAL loss 64.125000, w 0.500000
INITIAL loss 66.455887, w 0.500417
epoch 0, train loss 41.356121, w 1.179954
epoch 1, train loss 12.367311, w 1.551558
epoch 2, train loss 3.698371, w 1.754770
epoch 3, train loss 1.105976, w 1.865896
epoch 4, train loss 0.330736, w 1.926665
epoch 5, train loss 0.098904, w 1.959897
epoch 6, train loss 0.029577, w 1.978070
epoch 7, train loss 0.008845, w 1.988008
epoch 8, train loss 0.002645, w 1.993442
epoch 9, train loss 0.000791, w 1.996414
best w 1.996414
```
Results are replicable due to static seed.

## Theorical loss
Our lonely parameter $w$ starts as a random $[0, 1]$ float, so we could expect $w \approx 0.5$, thus, the MSE average loss is

$$\text{MSE loss} = \frac{\sum_{i=1}^{N} (y_{\text{pred}_i} - y_i)^2}{N}$$

where $N$ = TRAIN_SIZE, $y_{\text{pred}_i} = x_iw$ and $y_i = 2x_i$.

with $w = 0.5$, we have

$$\text{MSE loss} = \frac{\sum_{i=1}^{N} (0.5n_i - 2n_i)^2}{N}$$

$$= \frac{\sum_{i=1}^{N} (-1.5n_i)^2}{N}$$

$$= \frac{\sum_{i=1}^{N} 2.25n_i^2}{N}$$

$$= 2.25\frac{\sum_{i=1}^{N} n_i^2}{N}$$

Using the sum of squares formula $\sum_{i=1}^{N} i^2 = \frac{N(N+1)(2N+1)}{6}$, we have

$$\text{MSE loss} = 2.25\frac{N(N-1)(2N-1)}{6N}$$

$$= 2.25\frac{(N-1)(2N-1)}{6}$$

So, we could use this formula to compare the experimental loss with the theorical one.

### Running the experiment
Running M initializations with random $w$, we could get its average loss and w to compare with the theorical ones. This are the [results](#results).

## Training
For each hyperparameter epoch, we pass over the whole dataset and update the parameter $w$ on each example, using gradient descent.

How much do the loss change when we change $w$? We could use the partial derivative of the loss function with respect to $w$.

$$\text{Loss} = (y_{pred} - y)^2$$
$$\frac{\partial \text{Loss}}{\partial w} = 2(y_{pred} - y)\frac{\partial y_{pred}}{\partial w}$$
And given $y_{pred} = xw$, we have
$$= 2(y_{pred} - y)x$$

And the update w as

$$w_{new} = w_{old} - \alpha \frac{\partial \text{Loss}}{\partial w}$$

where $\alpha$ is the learning rate.

## Dataset generation
```c
#define TRAIN_SIZE 10

void genTrain(float train[][2], int size)
{
    for (int i = 0; i < size; ++i)
    {
        train[i][0] = (float)i;
        train[i][1] = 2 * train[i][0];
    }
    /*
    {0, 0}
    {1, 2}
    ...
    {N, 2N}
    */
}

int main()
{
    float train[TRAIN_SIZE][2];
    genTrain(train, TRAIN_SIZE);
    ...
}
```