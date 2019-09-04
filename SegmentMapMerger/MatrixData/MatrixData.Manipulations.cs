using System;
using System.Linq;

namespace MatrixOps
{
    public partial class MatrixData
    {
        public MatrixData SplitData(int rowStart, int colStart, int numRows = 0, int numCols = 0)
        {
            //This method will split the data based on the input and return it
            //it will also remove the data that was split from _data
            MatrixData splitData = new MatrixData(this, rowStart, colStart, numRows, numCols);

            int rowsToKeep = NumberOfRows - numRows;
            int colsToKeep = NumberOfColumns-numCols;
            int splitRowStart = rowStart + numRows;
            int splitColStart = colStart + numCols;
            TopSplit(splitRowStart,rowsToKeep);
            LeftSplit(splitColStart,colsToKeep);
            return splitData;
        }

        public MatrixData CopyData(int row = 0, int col = 0, int numRows = 0, int numCols = 0)
        {
            return new MatrixData(this, row, col, numRows, numCols);
        }

        public void AddRow(dynamic[] newRow)
        {
            if (newRow.Length != NumberOfColumns)
            {
                throw new Exception("Number of columns in the new row ("+newRow.Length+") must equal to the number of columns{"+NumberOfColumns+")");
            }
            dynamic[,] newData = new dynamic[NumberOfRows+1,NumberOfColumns];

            for (int row = 0; row < NumberOfRows; row++)
            {
                for (int col = 0; col <NumberOfColumns; col++)
                {
                    newData[row,col] = _data[row,col];
                }
            }
            for (int col = 0; col <NumberOfColumns; col++)
            {
                newData[NumberOfRows+1,col] = newRow[col];
            }
            _data = newData;
            NumberOfRows++;   
        }
        public void AddColumn(dynamic[] newCol = null,string headerName = "")
        {
            if (newCol.Length > NumberOfRows)
            {
                throw new Exception("Number of rows in the new column ("+newCol.Length+") must be less than or equal to the number of rows in the dataset{"+NumberOfRows+")");                
            }
            dynamic[] dataToAdd = new dynamic[NumberOfRows];
            for(int r = 0; r<newCol.Length; r++)
            {
                dataToAdd[r] = newCol[r];
            }
            dynamic[,] newData = new dynamic[NumberOfRows,NumberOfColumns+1];
            string[] newHeaders = new string[NumberOfColumns+1];
            for (int row = 0; row < NumberOfRows; row++)
            {
                for (int col = 0; col <NumberOfColumns; col++)
                {
                    newData[row,col] = _data[row,col];
                }
            }

            for (int col = 0; col < NumberOfColumns; col++)
            {
                newHeaders[col] = _headers[col];
            }
            newHeaders[NumberOfColumns] = (string.IsNullOrWhiteSpace(headerName))? NumberOfColumns.ToString() : headerName;

            for (int row = 0; row < NumberOfRows; row++)
            {
                newData[row,NumberOfColumns] = dataToAdd[row];
            }
            _data = newData;
            _headers = newHeaders;
            NumberOfColumns++;
        }
        public void ChangeRow(int row, dynamic[] newRow)
        {
            if (newRow.Length != NumberOfColumns)
            {
                throw new Exception("Number of columns in the new row ("+newRow.Length+") must equal to the number of columns{"+NumberOfColumns+")");
            }
            dynamic[][] data = _data.ToJagged();
            data[row] = newRow;
            _data = data.ToRectangular();
        }

        public void ChangeHeader(string[] colNames)
        {
            for (int i = 0; i < colNames.Length; i++)
            {
                ChangeHeader(i, colNames[i]);
            }
        }
        public void ChangeHeader(int col, string value)
        {
            _headers[col] = value;
        }
        public void ChangeRowHeader(string[] rowNames)
        {
            for (int i = 0; i < rowNames.Length; i++)
            {
                ChangeRowHeader(i, rowNames[i]);
            }
        }
        public void ChangeRowHeader(int row, string value)
        {
            if (_rowNames==null)
            {
                _rowNames = new string[NumberOfRows];
            }
            _rowNames[row] = value;
        }
        public void SetValue(int row, int col, dynamic value)
        {
            _data[row,col] = value;
        }
        public void SetValue(int row, int col, Func<dynamic> ValueFunction)
        {
            _data[row,col] = ValueFunction();
        }
        public void SetAll(dynamic value)
        {
            for(int row = 0; row < NumberOfRows; row++)
            {
                for (int col = 0; col < NumberOfColumns; col++)
                {
                    SetValue(row,col, value);
                }
            }
        }
        public void SetAll(Func<dynamic> ValueFunction)
        {
            for(int row = 0; row < NumberOfRows; row++)
            {
                for (int col = 0; col < NumberOfColumns; col++)
                {
                    SetValue(row,col, ValueFunction);
                }
            }
        }
        public void ReplaceIf(int row, int col, Func<double, bool> condition, double newVal)
        {
            _data[row,col] = condition(_data[row,col])? newVal : _data[row,col];
        }


        //https://stackoverflow.com/questions/30164019/shuffling-2d-array-of-cards        
        public void Suffle(int? seed = null)
        {
            Random random = seed==null? new Random() : new Random(seed.Value);
            
            dynamic[][] data = _data.ToJagged();

            for (int row = 0; row<NumberOfRows; row++)
            {
                dynamic[] currRow = data[row].Clone() as dynamic[];
                dynamic[] newRow = new dynamic[currRow.Length+1];
                newRow[0] = _rowNames[row];
                for (int col = 1; col<newRow.Length; col++)
                {
                    newRow[col] = currRow[col-1];
                }
                data[row] = newRow;             
            }
            
            data = data.OrderBy(t => random.Next()).ToArray();

            for (int row = 0; row<NumberOfRows; row++)
            {
                dynamic[] currRow = data[row].Clone() as dynamic[];
                dynamic[] newRow = new dynamic[currRow.Length-1];
                _rowNames[row] = currRow[0];
                for (int col = 1; col<currRow.Length; col++)
                {
                    newRow[col-1] = currRow[col];
                }           
                data[row] = newRow;       
            }

            _data = data.ToRectangular();
        }
        public void Transpose()
        {
            dynamic[,] output = new dynamic[NumberOfColumns, NumberOfRows];

            for (int row = 0; row < NumberOfRows; row++) 
            {
                for (int col = 0; col < NumberOfColumns; col++) 
                {
                    output[col, row] = _data[row, col];
                }
            }
            _data = output;
        }

        //Get the example pair for the matrix
        public MatrixData GetExemplar(int col, int numClasses, int startAt = 0, string colName = "")
        {
            int cols = NumberOfColumns + numClasses - 1;            
            MatrixData exemplarData = new MatrixData(NumberOfRows,cols);            
            for (int r = 0; r < NumberOfRows; r++)
            {
                double[] rr = new double[cols];
                int cc = 0;
                for (int c = 0; c < NumberOfColumns; c++)
                {
                    double d = _data[r,c];
                    if (c == col)
                    {
                        for (int j=0; j<numClasses; j++)
                        {
                            rr[cc] = 0;
                            if ((j)==((int)d) - startAt)
                            {
                                rr[cc] = 1;
                            }
                            cc++;
                        }
                    }
                    else
                    {
                        rr[cc] = d;
                        cc++;
                    }
                }
                exemplarData.ChangeRow(r, Array.ConvertAll<double,dynamic>(rr, x=> (dynamic)x));
                //exemplarData.ReSetColTypes();
            }
            exemplarData.CopyMetaData(this);
            string headerName = (string.IsNullOrWhiteSpace(colName))? "Exemplar " : colName;
            for (int i = 1; i<=numClasses;i++)
            {
                exemplarData.DetermineColType(cols-i);
                exemplarData.ChangeHeader(cols-i, headerName+(numClasses-i+1));
            }

            return exemplarData;
        }

        public dynamic[] GetVectorizedMatrix()
        {
            dynamic[] output = new dynamic[NumberOfRows*NumberOfColumns];
            int outputCounter = 0;
            for (int r = 0; r<NumberOfRows;r++)
            {
                for (int c = 0; c<NumberOfColumns;c++)
                {
                    output[outputCounter] = _data[r,c];
                    outputCounter++;
                }
            }
            return output;
        }
        
        public void Sort(int col, bool acen = true)
        {
            if(acen)
            {
                _data = _data.ToJagged().OrderBy(t => t[col]).ToArray().ToRectangular();
            }
            else
            {
                 _data = _data.ToJagged().OrderByDescending(t => t[col]).ToArray().ToRectangular();
            }
        }

        public void TopSplit(int rowStart, int numRowsToKeep)
        {
            dynamic[,] newData = new dynamic[numRowsToKeep,NumberOfColumns];
            string[] newRowNames = new string[numRowsToKeep];

            for (int row = rowStart; row < rowStart+numRowsToKeep; row++) 
            {
                for (int col = 0; col < NumberOfColumns; col++)
                {
                    newData[row-rowStart, col] = _data[row, col];
                }
                newRowNames[row-rowStart] = _rowNames[row];
            }
            _data = newData;
            _rowNames = newRowNames;
            NumberOfRows = numRowsToKeep;
        }

        public void LeftSplit(int colStart, int numColsToKeep)
        {
            dynamic[,] newData = new dynamic[NumberOfRows,numColsToKeep];
            string[] newHeaders = new string[numColsToKeep];

            Type[] newColumnDataTypes = new Type[numColsToKeep];
            for (int col = colStart; col < colStart+numColsToKeep; col++)
            {
                newHeaders[col-colStart] = _headers[col];
                newColumnDataTypes[col-colStart] = _columnDataTypes[col];
            }
            _headers = newHeaders;
            _columnDataTypes = newColumnDataTypes;

            for (int row = 0; row < NumberOfRows; row++) 
            {
                for (int col = colStart; col < colStart+numColsToKeep; col++)
                {
                    newData[row, col-colStart] = _data[row, col];
                }
            }
            _data = newData;
            NumberOfColumns = _headers.Length;
        }
    }
}