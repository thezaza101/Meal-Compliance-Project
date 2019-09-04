using System;
using System.Linq;
using System.Collections;

namespace MatrixOps
{
    public partial class MatrixData
    {
        public int Length
        {
            get{
                return _data.Length;
            }
        }
        public dynamic[] this[int row]
        {
            get
            {
                return _data.ToJagged()[row];
            }
        }
        public dynamic this[int row, int col]
        {
            get
            {
                return _data[row,col];
            }
            set{
                _data[row,col] = value;
            }
        }
        public dynamic[] Rows(int row)
        {
            return this[row];
        }

        public object[] Columns(string colName)
        {
            return Columns(Array.FindIndex(_headers,c => c.Equals(colName)));
        }
        public dynamic[] Columns(int col)
        {
            dynamic[] output = new dynamic[NumberOfRows];
            for (int row = 0; row<NumberOfRows;row++)
            {
                output[row] = _data[row,col];
            }   
            return output;
        }

        public T[] GetColumnCopy<T>(string colName)
        {
            return Array.ConvertAll<dynamic, T>(Columns(Array.FindIndex(_headers,c => c.Equals(colName))), x=> (T)x);
        }
        public T[] GetColumnCopy<T>(int col)
        {
            return Array.ConvertAll<dynamic, T>(Columns(col), x=> (T)x);
        }
        public MatrixData GetColumnCopy(string colName)
        {
            return GetColumnCopy(Array.FindIndex(_headers,c => c.Equals(colName)));
        }
        public MatrixData GetColumnCopy(int col)
        {
            return this.CopyData(0,col,0,1);
        }

        public int IndexOf(string colName)
        {
            return Array.FindIndex(_headers,c => c.Equals(colName));
        }
        public double GetValueOfCell(string column, string searchCol, string searchVal)
        {
            int indexOfReturnVal = _headers.ToList().IndexOf(_headers.Where(it => it == column).First());
            int indexOfSearchVal = _headers.ToList().IndexOf(_headers.Where(it => it == searchCol).First());
            var col = GetColumnCopy<string>(indexOfSearchVal).ToList();
            int indexOfValue = col.IndexOf(searchVal);
            return _data[indexOfValue,indexOfReturnVal];
        }
    }
}