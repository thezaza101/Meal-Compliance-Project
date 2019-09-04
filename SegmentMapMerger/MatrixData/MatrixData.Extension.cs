using System;
using System.Collections.Generic;
namespace MatrixOps
{
    public static class MatrixDataExtension
    {
        public static bool IsValueNumaric(this object o)
        {
            double d;
            return double.TryParse(o.ToString(), out d);
        }
        public static double ToDouble(this object o)
        {
            double d;
            double.TryParse(o.ToString(),out d);
            return d;
        }
        public static double ToInt32(this object o)
        {
            int i;
            int.TryParse(o.ToString(), out i);
            return i;
        }
        //Convert rectangular array to jagged array
        public static dynamic[][] ToJagged(this dynamic[,] array)
        {
            int height = array.GetLength(0);
            int width = array.GetLength(1);
            dynamic[][] jagged = new dynamic[height][];

            for (int i = 0; i < height; i++)
            {
                dynamic[] row = new dynamic[width];
                for (int j = 0; j < width; j++)
                {
                    row[j] = array[i, j];
                }
                jagged[i] = row;
            }
            return jagged;
        }

        //Convert jagged array to rectangular array
        public static dynamic[,] ToRectangular(this dynamic[][] array)
        {
            int height = array.Length;
            int width = array[0].Length;
            dynamic[,] rect = new dynamic[height, width];
            for (int i = 0; i < height; i++)
            {
                dynamic[] row = array[i];
                for (int j = 0; j < width; j++)
                {
                    rect[i, j] = row[j];
                }
            }
            return rect;
        }
        public static T[] GenerateEmptyArray<T>(int size,T value) 
        {
            T[] output = new T[size];

            for ( int i = 0; i < output.Length;i++ ) {
                output[i] = value;
            }
            return output;
        }
        public static dynamic[] ToDynamicArray(this double[] input)
        {
            return Array.ConvertAll<double, dynamic>(input, x=> (dynamic)x);
        }
        public static dynamic[][] ToDynamicArray(this double[][] input)
        {
            dynamic[][] output = new dynamic[input.Length][];
            for (int row = 0; row < input.Length;row++)
            {
                output[row] = Array.ConvertAll<double, dynamic>(input[row], x=> (dynamic)x);
            }
            return output;
        }
        public static dynamic[,] ToDynamicArray(this double[,] input)
        {
            dynamic[,] output = new dynamic[input.GetLength(0),input.GetLength(1)];
            for (int row = 0; row < input.GetLength(0);row++)
            {
                for (int col = 0; col< input.GetLength(1);col++)
                {
                    output[row,col] = input[row,col];
                }                
            }
            return output;
        }
        public static double[][] ToDoubleArray(this dynamic[][] input)
        {
            double[][] output = new double[input.Length][];
            for (int row = 0; row < input.Length;row++)
            {
                output[row] = Array.ConvertAll<dynamic, double>(input[row], x=> (double)x);
            }
            return output;
        }

        public static MatrixData GetTranspose(this MatrixData input)
        {
            MatrixData output = input.CopyData();
            output.Transpose();
            return output;
        }
        public static MatrixData GetReShapedMatrix(this dynamic[] input, int rowCount)
        {
            dynamic[,] outputArray = new dynamic[input.Length/rowCount,rowCount];
            int inputCounter = 0;
            for (int r = 0; r<outputArray.GetLength(0);r++)
            {
                for (int c = 0; c<outputArray.GetLength(1);c++)
                {
                    outputArray[r,c] = input[inputCounter];
                    inputCounter++;
                }
            }
            return new MatrixData(outputArray);
        }
        public static string[] Lines(this string input) => input.Split(new [] { '\r', '\n' });

        public static bool StringEquals(dynamic d, dynamic stringData) => string.Equals((string)d,(string)stringData);

        //https://stackoverflow.com/questions/14683467/finding-the-first-and-third-quartiles
        internal static double Percentile(this MatrixData input, double p)
        {

            double[] sortedData = Array.ConvertAll<dynamic, double>(input.GetVectorizedMatrix(), x=> (double)x);
            Array.Sort(sortedData);
            
            // algo derived from Aczel pg 15 bottom
            if (p >= 100.0d) return sortedData[sortedData.Length - 1];

            double position = (sortedData.Length + 1) * p / 100.0;
            double leftNumber = 0.0d, rightNumber = 0.0d;

            double n = p / 100.0d * (sortedData.Length - 1) + 1.0d;

            if (position >= 1)
            {
                leftNumber = sortedData[(int)Math.Floor(n) - 1];
                rightNumber = sortedData[(int)Math.Floor(n)];
            }
            else
            {
                leftNumber = sortedData[0]; // first data
                rightNumber = sortedData[1]; // first data
            }

            //if (leftNumber == rightNumber)
            if (Equals(leftNumber, rightNumber))
                return leftNumber;
            double part = n - Math.Floor(n);
            return leftNumber + part * (rightNumber - leftNumber);
        }
        
        
    }
}