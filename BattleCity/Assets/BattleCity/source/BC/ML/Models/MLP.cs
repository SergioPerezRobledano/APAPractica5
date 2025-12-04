
using System.Collections.Generic;
using UnityEngine;

public class MLPParameters
{
    List<float[,]> coeficients;
    List<float[]> intercepts;

    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }

    public List<float[,]> GetCoeff()
    {
        return coeficients;
    }
    public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }
    public List<float[]> GetInter()
    {
        return intercepts;
    }
}

public class MLPModel
{
    MLPParameters mlpParameters;
    public MLPModel(MLPParameters p)
    {
        mlpParameters = p;
    }

    public float[] FeedForward(float[] input)
    {
        // Obtenemos los pesos (coeficients) y sesgos (intercepts)
        List<float[,]> coefficients = mlpParameters.GetCoeff();
        List<float[]> intercepts = mlpParameters.GetInter();

        float[] currentInput = input;

        // Iteramos por cada capa de la red
        for (int i = 0; i < coefficients.Count; i++)
        {
            // Dimensiones: rows = entradas de esta capa, cols = salidas (neuronas) de esta capa
            int rows = coefficients[i].GetLength(0);
            int cols = coefficients[i].GetLength(1);

            float[] layerOutput = new float[cols];

            // Multiplicación de Matrices: (Input * Pesos) + Sesgo
            for (int c = 0; c < cols; c++)
            {
                float sum = 0f;
                for (int r = 0; r < rows; r++)
                {
                    // Es importante que el tamaño de currentInput coincida con rows
                    sum += currentInput[r] * coefficients[i][r, c];
                }

                // Sumamos el sesgo (bias) de la neurona 'c'
                sum += intercepts[i][c];

                // Aplicamos función de activación
                // Si NO es la última capa, usamos Sigmoide (capas ocultas)
                if (i < coefficients.Count - 1)
                {
                    layerOutput[c] = sigmoid(sum);
                }
                else
                {
                    // Si ES la última capa, dejamos el valor tal cual para pasarlo a Softmax
                    layerOutput[c] = sum;
                }
            }
            // La salida de esta capa es la entrada de la siguiente
            currentInput = layerOutput;
        }

        // Al final aplicamos Softmax para obtener probabilidades
        return SoftMax(currentInput);
    }

    private float sigmoid(float z)
    {
        // Fórmula sigmoide estándar
        return 1.0f / (1.0f + Mathf.Exp(-z));
    }

    public float[] SoftMax(float[] zArr)
    {
        float[] result = new float[zArr.Length];
        float sum = 0f;

        // 1. Calcular exponenciales y sumarlos
        for (int i = 0; i < zArr.Length; i++)
        {
            result[i] = Mathf.Exp(zArr[i]);
            sum += result[i];
        }

        // 2. Normalizar cada elemento
        for (int i = 0; i < zArr.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }

    public int Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        return index;
    }

    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        int index = 0;

        // Buscamos el índice con la probabilidad más alta
        for (int i = 1; i < output.Length; i++)
        {
            if (output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }
        return index;
    }
}
