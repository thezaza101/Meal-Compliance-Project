using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace MatrixOps
{
    public partial class MatrixData : ICloneable
    {
        //This represents the headers of the maxtrix information
        public string[] Headers { get => _headers; }
        public string[] RowNames { get => _rowNames; }
        public Type[] ColumnDataTypes {get => _columnDataTypes; }
        public dynamic[,] Data {get => _data; }

        public int NumberOfRows{get;private set;}
        public int NumberOfColumns{get;private set;}
        public Type DefaultNumericType {get; private set;} = typeof(double);

        private string[] _headers;
        private string[] _rowNames;
        private Type[] _columnDataTypes;
        private dynamic[,] _data;

        public MatrixData() {}
        
        public MatrixData(dynamic [,] data, Type defaultNumericType = null)
        {
            SetDefaultNumericType(defaultNumericType);
            _data = data;
            NumberOfRows = _data.GetLength(0);
            NumberOfColumns = data.GetLength(1);
            DetermineColTypes();      
            SetHeaders(new String(',', NumberOfColumns-1).ToString(),false,',');   
            SetRowNames();   
        }
        public MatrixData(dynamic [][] data, Type defaultNumericType = null) : this(data.ToRectangular(),defaultNumericType){}
        public MatrixData(dynamic [] data, Type defaultNumericType = null) : this (Make2DArray(data),defaultNumericType){}
        public MatrixData(int numRows, int numCols, Type defaultNumericType = null)
        {
            NumberOfRows = numRows;
            NumberOfColumns = numCols;              
            _columnDataTypes = MatrixDataExtension.GenerateEmptyArray<Type>(numCols,typeof(object));
            _data = Make2DArray(MatrixDataExtension.GenerateEmptyArray<dynamic>(numRows,default(double)),numCols);
            SetDefaultNumericType(defaultNumericType);
            SetHeaders(new String(',', numCols-1).ToString(),false,',');    
            SetRowNames();
        }
        public MatrixData(string input, bool hasHeaders = true, bool inputIsFile = true, char delimiter = ',', Type defaultNumericType = null)
        {
            SetDefaultNumericType(defaultNumericType);
            if(inputIsFile)
            {
                ReadFromCSV(input, hasHeaders,delimiter);
            }
            else
            {
                ReadFromString(input, hasHeaders,delimiter);
            }

            
        }

        public MatrixData(MatrixData dt, int rowStart, int colStart, int numRows = 0, int numCols = 0, Type defaultNumericType = null)
        {
            SetDefaultNumericType(defaultNumericType);
            _data = dt._data.Clone() as dynamic[,];
            _columnDataTypes = dt._columnDataTypes.Clone() as Type[];            
            _headers = dt._headers.Clone() as string[];
            _rowNames = dt._rowNames.Clone()  as string[];
            this.NumberOfColumns = dt.NumberOfColumns;
            this.NumberOfRows = dt.NumberOfRows;
            this.DefaultNumericType = dt.DefaultNumericType;
            int rowsToKeep = (numRows == 0)? (NumberOfRows - rowStart) : numRows;
            int colsToKeep = (numCols == 0)? (NumberOfColumns - colStart) : numCols;

            TopSplit(rowStart, rowsToKeep);
            LeftSplit(colStart, colsToKeep);
            
        }

        public MatrixData(System.Data.DataTable dt, Type defaultNumericType = null)
        {
            SetDefaultNumericType(defaultNumericType);
            int numRows = dt.Rows.Count;
            int numCols = dt.Columns.Count;
            _headers = MatrixDataExtension.GenerateEmptyArray<string>(numCols,"");
            _columnDataTypes = MatrixDataExtension.GenerateEmptyArray<Type>(numCols,typeof(object));
            _data = new dynamic[numRows,numCols];
            NumberOfRows = numRows;
            NumberOfColumns = numCols;
            dynamic[][] data = new dynamic[numRows][];
            dynamic[] row;
            for(int r = 0; r<numRows; r++)
            {
                row = new dynamic[numCols];
                for (int c = 0; c<numCols; c++)
                {
                    row[c] = dt.Rows[r][c];
                }
                data[r] = row;
            }
            _data = data.ToRectangular();
            for (int c = 0; c<numCols; c++)
            {
                _headers[c] = dt.Columns[c].ColumnName;
            }            
            DetermineColTypes();
            SetRowNames();
        }           

        private void SetDefaultNumericType(Type defaultNumericType)
        {
            if(!(defaultNumericType==null))
            {
                DefaultNumericType = defaultNumericType;
            }
        }
        public object Clone()
        {
            return this.MemberwiseClone();
        }
    }
}
