
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
    /// Parameters required for model input. By default it will be perception, kart position and time, 
    /// but depending on the data cleaning and data acquisition modificiations made by each one, the input will need more parameters.
    /// </summary>
    /// <param name="p">The Agent perception</param>
    /// <returns>The action label</returns>
    public float[] FeedForward(float[] input)
    {
        // Obtenemos coeficientes (pesos) y los intercepts (sesgos)
        List<float[,]> coefficients = mlpParameters.GetCoeff();
        List<float[]> intercepts = mlpParameters.GetInter();

        float[] currentInput = input;

        // Iteramos por cada capa de la red
        for (int i = 0; i < coefficients.Count; i++)
        {
            int rows = coefficients[i].GetLength(0); // Neuronas capa anterior (input dim)
            int cols = coefficients[i].GetLength(1); // Neuronas capa actual (output dim)

            float[] layerOutput = new float[cols];

            // Multiplicación de Matrices: Input * Pesos + Sesgo
            for (int c = 0; c < cols; c++)
            {
                float sum = 0f;
                for (int r = 0; r < rows; r++)
                {
                    // currentInput debe coincidir con rows
                    sum += currentInput[r] * coefficients[i][r, c];
                }
                // Sumar el sesgo (intercept)
                sum += intercepts[i][c];

                // Aplicar función de activación
                // Si NO es la última capa, aplicamos Sigmoide (capas ocultas)
                // Si ES la última capa, preparamos para Softmax (output layer)
                if (i < coefficients.Count - 1)
                {
                    layerOutput[c] = sigmoid(sum);
                }
                else
                {
                    // En la última capa pasamos el valor crudo (logits) a la SoftMax
                    layerOutput[c] = sum;
                }
            }
            // La salida de esta capa se convierte en la entrada de la siguiente
            currentInput = layerOutput;
        }

        // Aplicamos SoftMax al resultado final para obtener probabilidades
        return SoftMax(currentInput);
    }

    private float sigmoid(float z)
    {
        // Fórmula Sigmoide: 1 / (1 + e^-z)
        return 1.0f / (1.0f + Mathf.Exp(-z));
    }

    public float[] SoftMax(float[] zArr)
    {
        float[] result = new float[zArr.Length];
        float sum = 0f;

        // 1. Calcular exponenciales y la suma total
        for (int i = 0; i < zArr.Length; i++)
        {
            result[i] = Mathf.Exp(zArr[i]);
            sum += result[i];
        }

        // 2. Normalizar dividiendo por la suma
        for (int i = 0; i < zArr.Length; i++)
        {
            result[i] = result[i] / sum;
        }

        return result;
    }

    

    /// <summary>
    /// Elige el output de mayor nivel
    /// </summary>
    /// <param name="output"></param>
    /// <returns></returns>
    public int Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        return index;
    }

    /// <summary>
    /// Obtiene el índice de mayor valor.
    /// </summary>
    /// <param name="output"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        int index = 0;

        // Buscar el índice con la mayor probabilidad
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
