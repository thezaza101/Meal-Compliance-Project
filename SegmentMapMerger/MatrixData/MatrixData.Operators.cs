using System;

namespace MatrixOps
{
    public partial class MatrixData
    {
        public static MatrixData operator +(MatrixData a, MatrixData b)
        {
            if (a.Length != b.Length)
            {
                throw new InvalidOperationException("Matrices of different length cannot be added");
            }
            MatrixData output = a.CopyData(0, 0);
            for (int r = 0; r < a.NumberOfRows; r++)
            {
                for (int c = 0; c < a.NumberOfColumns; c++)
                {
                    output[r, c] = output[r, c] + b[r, c];
                }
            }
            return output;
        }
        public static MatrixData operator -(MatrixData a, MatrixData b)
        {
            if (a.Length != b.Length)
            {
                throw new InvalidOperationException("Matrices of different length cannot be subtracted");
            }
            MatrixData output = a.CopyData(0, 0);
            for (int r = 0; r < a.NumberOfRows; r++)
            {
                for (int c = 0; c < a.NumberOfColumns; c++)
                {
                    output[r, c] = output[r, c] - b[r, c];
                }
            }
            return output;
        }

        public static MatrixData operator *(MatrixData a, MatrixData b)
        {
            if (a.NumberOfRows == b.NumberOfRows && a.NumberOfColumns == b.NumberOfColumns)
            {
                double[,] m = new double[a.NumberOfRows, a.NumberOfColumns];

                for(int row = 0; row< a.NumberOfRows; row++)
                {
                    for (var col = 0; col < a.NumberOfColumns; col++)
                    {
                        m[row,col] = a[row,col] * b[row,col];
                    }
                }
                return new MatrixData(m.ToDynamicArray());
            }

            var newMatrix = new double[a.NumberOfRows, b.NumberOfColumns];

            if (a.NumberOfColumns == b.NumberOfRows)
            {
                for(int i = 0; i<a.NumberOfRows; i++)
                {
                    for (int j = 0; j < b.NumberOfColumns; j++)
                    {
                        double temp = 0.0;

                        for (int k = 0; k < a.NumberOfColumns; k++)
                        {
                            temp += a[i,k] * b[k,j];
                        }

                        newMatrix[i,j] = temp;
                    }
                }
            }

            return new MatrixData(newMatrix.ToDynamicArray());
        }
        public static MatrixData operator /(MatrixData a, MatrixData b)
        {
            throw new NotImplementedException();
        }



        public static MatrixData operator +(MatrixData a, double b)
        {
            MatrixData output = a.CopyData(0, 0);
            for (int r = 0; r < a.NumberOfRows; r++)
            {
                for (int c = 0; c < a.NumberOfColumns; c++)
                {
                    output[r, c] = output[r, c] + b;
                }
            }
            return output;
        }
        public static MatrixData operator -(MatrixData a, double b)
        {
            MatrixData output = a.CopyData(0, 0);
            for (int r = 0; r < a.NumberOfRows; r++)
            {
                for (int c = 0; c < a.NumberOfColumns; c++)
                {
                    output[r, c] = output[r, c] - b;
                }
            }
            return output;
        }
        public static MatrixData operator *(MatrixData a, double b)
        {
            MatrixData output = a.CopyData(0, 0);
            for (int r = 0; r < a.NumberOfRows; r++)
            {
                for (int c = 0; c < a.NumberOfColumns; c++)
                {
                    output[r, c] = output[r, c] * b;
                }
            }
            return output;
        }
        public static MatrixData operator /(MatrixData a, double b)
        {
            MatrixData output = a.CopyData(0, 0);
            for (int r = 0; r < a.NumberOfRows; r++)
            {
                for (int c = 0; c < a.NumberOfColumns; c++)
                {
                    output[r, c] = output[r, c] / b;
                }
            }
            return output;
        }
    }
}