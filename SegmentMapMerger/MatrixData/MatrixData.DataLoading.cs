using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace MatrixOps
{
    public partial class MatrixData
    {
        //This method reads a CSV file and sets the data.
        public void ReadFromCSV(string filelocation, bool hasHeaders, char delimiter)
        {
            string[] lines = ReadAllLines(filelocation);
            ProcessLines(lines,hasHeaders,delimiter);
           
        }

        public void ReadFromString(string input, bool hasHeaders, char delimiter)
        {
            ProcessLines(input.Lines(), hasHeaders, delimiter);
        }

        private void ProcessLines(string[] lines,bool hasHeaders, char delimiter)
        {
             int rowStartIndex = (hasHeaders)? 1 : 0;

            NumberOfRows = lines.Length-rowStartIndex;
            SetHeaders(lines[0], hasHeaders,delimiter);
            NumberOfColumns = _headers.Length;

            _data = new dynamic[NumberOfRows,NumberOfColumns];
            
            for (int row = rowStartIndex; row<lines.Length; row++)
            {
                string[] data = SplitCsvLine(lines[row],delimiter);
                for(int col = 0; col < NumberOfColumns; col++)
                {
                    _data[row-rowStartIndex,col] = data[col];
                }
            }
            DetermineColTypes();

            for(int col = 0; col < NumberOfColumns; col++)
            {
                if (IsValueNumaric(col))
                {
                    for (int row = 0; row<NumberOfRows; row++)
                    {      
                        _data[row,col] = ConvertToNumeric(_data[row,col].ToString());
                    }
                }
                
            }
        }

        public void WriteCSV(string path, bool includeHeaders = true)
        {
            using (StreamWriter sw = new StreamWriter(path))
            {
                if(includeHeaders)
                {
                    sw.WriteLine(CreateHeaderRow());
                }
                dynamic[][] data = _data.ToJagged();
                for (int row = 0; row<NumberOfRows;row++)
                {
                    sw.WriteLine(CreateCSVRow(data[row]));
                }                
            }
        }

        private string CreateHeaderRow()
        {
            return CreateCSVRow(_headers);
        }
        
        private string CreateCSVRow(dynamic[] data)
        {
            string output = "";
            for (int col = 0; col < NumberOfColumns; col++)
            {
                output += data[col];
                if (col != (NumberOfColumns-1))
                {
                    output += ',';
                }
            }
            return output;
        }

        public void DetermineColTypes()
        {
            _columnDataTypes = new Type[NumberOfColumns];
            for (int col = 0; col < NumberOfColumns; col++)
            {
                _columnDataTypes[col] = GetColType(col);
            }
        }
        public void DetermineColType(int col)
        {
            _columnDataTypes[col] = GetColType(col);
        }
        private Type GetColType(int col)
        {
            bool isValueNumaric = true;
            try
            {
                Random r = new Random();
                for (int row = 0; row < NumberOfRows/10; row++)
                {
                    
                    var o = ConvertToNumeric(_data[r.Next(row, NumberOfRows),col].ToString());
                }
            } 
            catch (Exception)
            {
                isValueNumaric = false;
            }
            return (isValueNumaric)? typeof(double) : typeof(string);            
        }

        //This set the header values if the file has headers
        private void SetHeaders(string headersLine, bool hasHeaders, char delimiter)
        {
            if (hasHeaders)
            {
                _headers = SplitCsvLine(headersLine,delimiter);
            }
            else
            {
                int numCols = SplitCsvLine(headersLine,delimiter).Length;
                _headers = new string[numCols];
                for (int i = 0; i<numCols;i++)
                {
                    _headers[i] = (i).ToString();
                }
                //_headers = Enumerable.Repeat(string.Empty, SplitCsvLine(headersLine,delimiter).Length).ToArray();
            }         
            SetRowNames();
        }

        private void SetRowNames()
        {
            if (_rowNames==null)
            {
                _rowNames = new string[NumberOfRows];
                for (int i = 0; i<NumberOfRows;i++)
                {
                    _rowNames[i] = (i).ToString();
                }
            }
        }

        //https://stackoverflow.com/questions/12744725/how-do-i-perform-file-readalllines-on-a-file-that-is-also-open-in-excel
        //This method will read the CSV file and split it into lines
        private string[] ReadAllLines(string path)
        {
            using (var csv = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            {
                using (var sr = new StreamReader(csv))
                {
                    List<string> file = new List<string>();
                    while (!sr.EndOfStream)
                    {
                        file.Add(sr.ReadLine());
                    }

                    return file.ToArray();
                }
            }
        }

        //https://stackoverflow.com/questions/17207269/how-to-properly-split-a-csv-using-c-sharp-split-function
        //This method will safely split a line of a CSV file
        private string[] SplitCsvLine(string s,char delimiter) 
        {            
            //string pattern = @"""\s*,\s*""";
            //string pattern = @"""\s*"+delimiter+@"\s*""";

            // input.Substring(1, input.Length - 2) removes the first and last " from the string
            //string[] tokens = System.Text.RegularExpressions.Regex.Split(s.Substring(1, s.Length - 2), pattern);
            //return tokens;

            return s.Split(delimiter);            
        }
    }
}
