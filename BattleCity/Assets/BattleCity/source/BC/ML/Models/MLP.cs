using System;
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

    /// <summary>
    /// Feedforward del MLP.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public float[] FeedForward(float[] input)
    {
        List<float[,]> weights = mlpParameters.GetCoeff();
        List<float[]> biases = mlpParameters.GetInter();

        float[] activation = input;

        int numLayers = weights.Count;
        for (int l = 0; l < numLayers; l++)
        {
            int rows = weights[l].GetLength(0); // Número de neuronas en capa l+1
            int cols = weights[l].GetLength(1); // Número de neuronas en capa l

            // Depuración para verificar dimensiones
            Debug.Log($"Layer {l}: Rows = {rows}, Cols = {cols}");
            Debug.Log($"Layer {l}: Activation Length = {activation.Length}");

            // Verificación de que la activación tiene el tamaño correcto
            if (activation.Length != cols)
            {
                throw new Exception($"El tamaño de la activación no coincide con el número de columnas de los pesos en la capa {l}. " +
                                    $"Activación tiene longitud {activation.Length} pero los pesos tienen {cols} columnas.");
            }

            float[] z = new float[rows];

            // z = W * activation + b
            for (int i = 0; i < rows; i++)
            {
                z[i] = biases[l][i]; // sesgo para la neurona i
                for (int j = 0; j < cols; j++)
                {
                    z[i] += weights[l][i, j] * activation[j]; // Calculamos la suma ponderada
                }
            }

            // Activación
            if (l < numLayers - 1)
            {
                // Capas ocultas: sigmoide
                for (int i = 0; i < z.Length; i++)
                {
                    z[i] = sigmoid(z[i]);
                }
            }
            else
            {
                // Última capa: softmax
                z = SoftMax(z);
            }

            // Actualizar la activación para la siguiente capa
            activation = z;
        }

        return activation;
    }


    /// <summary>
    /// Sigmoide
    /// </summary>
    private float sigmoid(float z)
    {
        return 1f / (1f + Mathf.Exp(-z));
    }

    /// <summary>
    /// Softmax
    /// </summary>
    public float[] SoftMax(float[] zArr)
    {
        float max = float.NegativeInfinity;
        foreach (float v in zArr)
            if (v > max) max = v;

        float sumExp = 0f;
        float[] expArr = new float[zArr.Length];
        for (int i = 0; i < zArr.Length; i++)
        {
            expArr[i] = Mathf.Exp(zArr[i] - max); // estabilidad numérica
            sumExp += expArr[i];
        }

        for (int i = 0; i < expArr.Length; i++)
        {
            expArr[i] /= sumExp;
        }

        return expArr;
    }

    /// <summary>
    /// Predicción: índice de mayor salida
    /// </summary>
    public int Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        return index;
    }

    /// <summary>
    /// Obtener índice de mayor valor
    /// </summary>
    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        int index = 0;

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
