using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace MatrixOps
{
    public partial class MatrixData
    {
        public bool LabeledRows {get;set;} = false;
        public MatrixData Head(int numberOfRows = 5)
        {
            return new MatrixData(this,0,0,numberOfRows,0);
        }
        public MatrixData Tail(int numberOfRows = 5)
        {
            return new MatrixData(this,NumberOfRows-numberOfRows,0,numberOfRows,0);
        }

        public string GetRowName(int index)
        {
            return _rowNames[index];
        }

        private static T[,] Make2DArray<T>(T[] input, int numcols =1)
        {
            T[,] output = new T[input.Length, numcols];
            for (int row = 0; row < input.Length; row++)
            {
                for (int col = 0; col<numcols;col++)
                {
                    output[row,col] = input[row];
                }
            }
            return output;
        }
        public void CopyMetaData(MatrixData data)
        {
            var copiedHeaders = data._headers.Clone() as string[];
            var copiedTypes =  data._columnDataTypes.Clone() as Type[];
            try
            {
                for(int i = 0; i<NumberOfColumns;i++)
                {
                    this._headers[i] = copiedHeaders[i];
                    this._columnDataTypes[i] = copiedTypes[i];
                }
            }
            catch
            {

            }
        }
        
        //https://stackoverflow.com/questions/33417721/convert-a-object-array-into-a-dataset-datatable-in-c-sharp
        public System.Data.DataTable ToDataTable()
        {
            System.Data.DataTable dt = new System.Data.DataTable();
            for (int col = 0; col < _data.GetLength(1); col++)
            {
                dt.Columns.Add(_headers[col]);
            }

            for (var row = 0; row < _data.GetLength(0); ++row)
            {
                System.Data.DataRow r = dt.NewRow();
                for (var col = 0; col < _data.GetLength(1); ++col)
                {
                    r[col] = _data[row, col];
                }
                dt.Rows.Add(r);
            }
            return dt;
        }

        public override string ToString()
        {
            return ToString(NumberOfRows,10,300,true);
        }
        public string ToString(int colWidth)
        {
            return ToString(NumberOfRows,colWidth,300,true);
        }

        public string ToString(int numberOfRows = 5, int colWidth = 10, int maxRows = 300, bool? printRowLabels = null, bool printDataTypes = false)
        {
            string output ="";
            bool printRowLabs = (printRowLabels == null)? false : true;
            printRowLabs = (printRowLabs|LabeledRows)? true:false;

            numberOfRows = (numberOfRows > maxRows)? maxRows : numberOfRows;
            var row1 = (NumberOfColumns*colWidth);
            var row2 = (Convert.ToInt32(printRowLabs)*colWidth);
            int rowWidth = (NumberOfColumns*colWidth)+(Convert.ToInt32(printRowLabs)*colWidth);

            output += (printRowLabs)? ColValue(NumberOfRows.ToString()+"x"+NumberOfColumns.ToString()) + '|' : "";
            
            
                foreach (string h in _headers)
                {
                    output += ColValue(h) + '|';
                }
                output += Environment.NewLine;

                
            

            if (printDataTypes)
            {
                output += (printRowLabs)? ColValue("...") + '|' : "";
                foreach (Type t in _columnDataTypes)
                {
                    output += ColValue(t.Name,'-') + '|';
                }
                output += Environment.NewLine;
            }

            output += new String('-', rowWidth-(rowWidth/10)).ToString();
            
            for (int r = 0; r<numberOfRows;r++)
            {
                output += Environment.NewLine;
                output += MakeRow(r);
                
            }

            string MakeRow(int row)
            {
                string outRow = "";                
                if (printRowLabs)
                {
                    outRow += ColValue(_rowNames[row].ToString()) + '|';
                }

                for (int col = 0; col < NumberOfColumns; col++)
                {
                    outRow += ColValue(_data[row,col].ToString()) + '|';
                }
                return outRow;
            }

            string ColValue(string value, char fill = ' ')
            {
                int valLength = value.Length;
                string valueToWrite = "";
                if (valLength > colWidth-4)
                {
                    valueToWrite = value.Substring(0,colWidth-4);
                }
                else
                {
                    if(valLength%2==0)
                    {
                        int numspaces = ((colWidth-4) - valLength)/2;
                        valueToWrite = (new String(fill, numspaces)) + value + (new String(fill, numspaces));
                    }
                    else
                    {
                        int numspaces = ((colWidth-4) - valLength-1)/2;
                        valueToWrite = (new String(' ', numspaces)) + value + (new String(' ', numspaces))+" ";
                    }                    
                }
                return " "+valueToWrite+" ";
            }

            
            return output;
        }

        private List<Type> numaricTypes = new List<Type>{typeof(double),typeof(int),typeof(decimal)};
        //Determines if the input value is numaric
        private bool IsValueNumaric(int col)
        {
            return numaricTypes.Contains(_columnDataTypes[col]);
        }
        
        //https://stackoverflow.com/questions/2961656/generic-tryparse
        //This method will try parse the string data to the Type specified when the class was created
        private dynamic ConvertToNumeric(string input)
        {
            var converter = TypeDescriptor.GetConverter(DefaultNumericType);
            if (converter != null)
            {
                return converter.ConvertFromString(input);
            }
            return null;
        }
    }
}