#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TRAIN_SIZE 10
#define EXPS 1000 // iteraciones para promediar LOSS y W iniciales y comparar con teóricos

// Hiperparámetros
#define LR 0.001
#define EPOCHS 10

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

float rand_float()
{
    return (float)rand() / RAND_MAX;
}

float MSE(float y_pred, float y)
{
    return (y_pred - y) * (y_pred - y);
}

int main()
{
    float train[TRAIN_SIZE][2];
    genTrain(train, TRAIN_SIZE);

    // Modelo de regresión lineal y = x*w (w should be 2)
    // float w = rand_float();

    // THEORICAL LOSS
    // w = 0.5
    // loss = sum( (y_pred - y)^2 ) / N
    // => sum( ((0 1 ... (N-1)) * 0.5 - (0 2 ... 2N))^2 ) / N
    // => sum( ((0 0.5 ... (N-1)/2) - (0 2 ... 2N))^2 ) / N
    // => sum( (0 -1.5 ... -1.5N)^2 ) / N
    // => sum(0 2.25 ... 2.25N^2) / N
    // => 2.25 * sum(0 1 ... N^2) / N
    // sumatoria de cuadrados = n(n+1)(2n+1)/6
    // => 2.25 * (N-1) * N * (2(N-1)+1) / 6N
    // => 2.25 * (N-1) * (2N-1) / 6
    float loss_sum = 0;
    float w_sum = 0;
    for (int exp = 0; exp < EXPS; exp++)
    {
        float w = rand_float();
        w_sum += w;

        float loss = 0;
        for (int i = 0; i < TRAIN_SIZE; i++)
        {
            float x = train[i][0];
            float y = train[i][1];

            float y_pred = x * w;
            loss += MSE(y_pred, y);
        }
        loss_sum += loss / TRAIN_SIZE;
    }
    printf("TRAIN_SIZE = %d\n", TRAIN_SIZE);
    printf("THEORICAL loss %f, w %f\n", 2.25 * (TRAIN_SIZE - 1) * (2 * TRAIN_SIZE - 1) / 6, 0.5);
    float w = w_sum / EXPS; // w inicial para entrenar
    printf("INITIAL loss %f, w %f\n", loss_sum / EXPS, w);

    // TRAIN
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        float epoch_loss = 0;
        for (int i = 0; i < TRAIN_SIZE; i++)
        {
            float x = train[i][0];
            float y = train[i][1];

            float y_pred = x * w;
            float loss = MSE(y_pred, y);
            epoch_loss += loss;

            // ¿Cómo cambia la pérdida con w? : derivada parcial de MSE con respecto a w
            float grad = 2 * (y_pred - y) * x;

            w -= LR * grad;
        }
        epoch_loss /= TRAIN_SIZE;
        printf("epoch %d, train loss %f, w %f\n", epoch, epoch_loss, w);
    }

    printf("best w %f\n", w);
    return 0;
}